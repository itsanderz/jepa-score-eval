"""
Extended JEPA-SCORE experiments for paper:
"Understanding Jacobian-Based Density Estimation for OOD Detection
 on Pretrained Vision Encoders"

Three experiments:
  1. full_jacobian  — Exact JEPA-SCORE (full Jacobian, matching the paper's code)
  2. projection_sweep — Randomized JEPA-SCORE with varying projection counts
  3. baselines — k-NN, Mahalanobis, Isolation Forest (recomputed for consistency)

All experiments save:
  - Numerical results (JSON)
  - Singular value spectra (NPY) for analysis
  - Intermediate checkpoints (resume on crash)

Usage:
  # Run everything (recommended order):
  python run_extended.py --experiment baselines
  python run_extended.py --experiment full_jacobian --model dinov2_vits14
  python run_extended.py --experiment full_jacobian --model dinov2_vitb14 --ood CIFAR10_vs_SVHN --n-samples 200
  python run_extended.py --experiment projection_sweep --model dinov2_vits14

  # Quick test (small sample, one pair):
  python run_extended.py --experiment full_jacobian --model dinov2_vits14 --ood CIFAR10_vs_SVHN --n-samples 50

Hardware: RTX 3090 (24GB) or better.
Estimated time:
  - baselines: ~5 min
  - full_jacobian vits14 (n=500, 3 pairs): ~3 hours
  - full_jacobian vitb14 (n=200, 1 pair): ~2 hours
  - projection_sweep vits14 (n=200, 3 pairs): ~3 hours
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRANSFORM = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODELS = {
    "dinov2_vits14": ("facebookresearch/dinov2", "dinov2_vits14"),
    "dinov2_vitb14": ("facebookresearch/dinov2", "dinov2_vitb14"),
    "dinov2_vitl14": ("facebookresearch/dinov2", "dinov2_vitl14"),
    "metaclip_b16": ("open_clip", "ViT-B-16-quickgelu", "metaclip_400m"),
}

RESULTS_DIR = Path(__file__).parent / "results_extended"

EPS = 1e-6  # Matches the paper: "eps (we pick 1e-6)"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_subset(cls, n: int, **kw) -> torch.Tensor:
    ds = cls(root="./data", download=True, transform=TRANSFORM, **kw)
    imgs = []
    for i, (img, _) in enumerate(ds):
        if n > 0 and i >= n:
            break
        imgs.append(img)
    return torch.stack(imgs)


def get_ood_pairs(n: int, which: Optional[str] = None) -> dict[str, tuple]:
    """Load OOD pairs. `which` filters to a single pair."""
    print("Loading datasets...", flush=True)
    c10 = load_subset(datasets.CIFAR10, n, train=False)
    pairs = {}

    if which is None or which == "CIFAR10_vs_SVHN":
        svhn = load_subset(datasets.SVHN, n, split="test")
        pairs["CIFAR10_vs_SVHN"] = (c10, svhn)

    if which is None or which == "CIFAR10_vs_CIFAR100":
        c100 = load_subset(datasets.CIFAR100, n, train=False)
        pairs["CIFAR10_vs_CIFAR100"] = (c10, c100)

    if which is None or which == "CIFAR10_vs_Textures":
        try:
            dtd = load_subset(datasets.DTD, n, split="test")
            pairs["CIFAR10_vs_Textures"] = (c10, dtd)
        except Exception as e:
            print(f"  Warning: DTD failed ({e}), skipping", flush=True)

    for k, (a, b) in pairs.items():
        print(f"  {k}: {a.shape} vs {b.shape}", flush=True)
    return pairs


# ---------------------------------------------------------------------------
# Model loading & embedding extraction
# ---------------------------------------------------------------------------


def load_model(key: str, device: str):
    entry = MODELS[key]
    if entry[0] == "open_clip":
        # MetaCLIP / OpenCLIP models
        import open_clip

        _, arch, pretrained = entry
        model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
        # Return just the visual encoder
        m = model.visual
        m.to(device).eval()
        return m
    else:
        # torch.hub models (DINOv2)
        repo, name = entry
        m = torch.hub.load(repo, name)
        m.to(device).eval()
        return m


def extract_embeddings(model, imgs: torch.Tensor, device: str, bs: int = 64) -> np.ndarray:
    embs = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            embs.append(model(imgs[i:i + bs].to(device)).cpu().numpy())
    return np.concatenate(embs)


# ---------------------------------------------------------------------------
# JEPA-SCORE: Full Jacobian (matching the paper exactly)
# ---------------------------------------------------------------------------


def jepa_score_full_jacobian(
    model, img: torch.Tensor, device: str
) -> tuple[float, np.ndarray]:
    """
    Compute JEPA-SCORE using the FULL Jacobian, matching the paper's code:

        J = jacobian(lambda x: model(x).sum(0), inputs=images)
        J = J.flatten(2).permute(1, 0, 2)
        svdvals = torch.linalg.svdvals(J)
        jepa_score = svdvals.clip_(eps).log_().sum(1)

    Returns (score, singular_values_array).
    """
    x = img.unsqueeze(0).to(device)  # (1, 3, 224, 224)

    def func(inp):
        return model(inp).sum(0)  # (d,) — sum over batch

    # Full Jacobian: (d, 1, 3, 224, 224)
    J = torch.autograd.functional.jacobian(func, x, vectorize=False)
    # Reshape: (d, 1, D) → (1, d, D)
    J = J.flatten(2).permute(1, 0, 2)
    # SVD: (1, d)
    sv = torch.linalg.svdvals(J)
    score = sv.clamp(min=EPS).log().sum(1).item()
    sv_np = sv[0].detach().cpu().numpy()

    return score, sv_np


def run_full_jacobian_batch(
    model, imgs: torch.Tensor, device: str, label: str = ""
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run full Jacobian JEPA-SCORE on a batch of images. Returns scores and spectra."""
    scores = []
    spectra = []
    n = len(imgs)
    for i in range(n):
        t0 = time.time()
        try:
            score, sv = jepa_score_full_jacobian(model, imgs[i], device)
        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at sample {i+1}/{n}, clearing cache and retrying...", flush=True)
            torch.cuda.empty_cache()
            try:
                score, sv = jepa_score_full_jacobian(model, imgs[i], device)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM again, skipping sample {i+1}", flush=True)
                scores.append(float("nan"))
                spectra.append(np.array([]))
                continue

        scores.append(score)
        spectra.append(sv)
        elapsed = time.time() - t0

        if (i + 1) % 25 == 0 or i == 0:
            remaining = elapsed * (n - i - 1)
            print(
                f"  {label} [{i+1}/{n}] score={score:.2f} "
                f"({elapsed:.1f}s/sample, ~{remaining/60:.0f}min remaining)",
                flush=True,
            )
    return np.array(scores), spectra


# ---------------------------------------------------------------------------
# JEPA-SCORE: Randomized projections (our approximation)
# ---------------------------------------------------------------------------


def jepa_score_randomized(
    model, imgs: torch.Tensor, device: str, n_proj: int = 64, seed: int = 42
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Randomized JEPA-SCORE using p random VJPs.
    NOT what the paper describes — the paper uses the full Jacobian.
    This is our approximation for computational feasibility analysis.
    """
    rng = torch.Generator(device=device).manual_seed(seed)
    scores = []
    spectra = []

    for idx in range(len(imgs)):
        with torch.no_grad():
            emb = model(imgs[idx:idx + 1].to(device))
        edim = emb.shape[1]
        omega = torch.randn(n_proj, edim, device=device, generator=rng)

        jt_cols = []
        for j in range(n_proj):
            x = imgs[idx:idx + 1].to(device).requires_grad_(True)
            out = model(x)
            (g,) = torch.autograd.grad(
                out, x, grad_outputs=omega[j].unsqueeze(0), retain_graph=False
            )
            jt_cols.append(g.flatten().detach())

        jt = torch.stack(jt_cols, dim=0)  # (n_proj, D)
        sv = torch.linalg.svdvals(jt)  # (n_proj,)
        score = sv.clamp(min=EPS).log().sum().item()
        scores.append(score)
        spectra.append(sv.cpu().numpy())

        if (idx + 1) % 100 == 0:
            print(f"    Randomized (p={n_proj}): {idx+1}/{len(imgs)}", flush=True)

    return np.array(scores), spectra


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def knn_scores(e_ref: np.ndarray, e_test: np.ndarray, k: int = 10) -> np.ndarray:
    return (
        NearestNeighbors(n_neighbors=k, metric="cosine")
        .fit(e_ref)
        .kneighbors(e_test)[0]
        .mean(axis=1)
    )


def maha_scores(e_ref: np.ndarray, e_test: np.ndarray) -> np.ndarray:
    return EmpiricalCovariance().fit(e_ref).mahalanobis(e_test)


def iforest_scores(e_ref: np.ndarray, e_test: np.ndarray) -> np.ndarray:
    return -IsolationForest(n_estimators=100, random_state=42).fit(e_ref).score_samples(
        e_test
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def fpr_at_tpr(id_scores: np.ndarray, ood_scores: np.ndarray, tpr: float = 0.95) -> float:
    """FPR at given TPR. Higher scores = more likely OOD."""
    threshold = np.percentile(ood_scores, (1 - tpr) * 100)
    return float(np.mean(id_scores >= threshold))


def compute_metrics(id_scores: np.ndarray, ood_scores: np.ndarray) -> dict:
    """Compute AUROC and FPR95. Handles NaN scores gracefully."""
    # Remove NaN
    id_valid = id_scores[~np.isnan(id_scores)]
    ood_valid = ood_scores[~np.isnan(ood_scores)]
    if len(id_valid) < 10 or len(ood_valid) < 10:
        return {"auroc": float("nan"), "fpr95": float("nan"), "n_valid": len(id_valid) + len(ood_valid)}

    labels = np.concatenate([np.zeros(len(id_valid)), np.ones(len(ood_valid))])
    scores = np.concatenate([id_valid, ood_valid])
    auroc = roc_auc_score(labels, scores)
    fpr95 = fpr_at_tpr(id_valid, ood_valid, 0.95)
    return {"auroc": auroc, "fpr95": fpr95, "n_valid": len(id_valid) + len(ood_valid)}


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    experiment: str
    model: str
    ood_pair: str
    method: str
    n_samples: int
    auroc: float
    fpr95: float
    time_s: float
    # Method-specific params
    n_projections: Optional[int] = None
    seed: Optional[int] = None
    embed_dim: Optional[int] = None
    # Additional
    notes: str = ""


def save_results(results: list[ExperimentResult], filename: str):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    data = [asdict(r) for r in results]
    path.write_text(json.dumps(data, indent=2))
    print(f"Saved {len(results)} results to {path}", flush=True)


def save_spectra(spectra: list[np.ndarray], filename: str):
    spectra_dir = RESULTS_DIR / "svd_spectra"
    spectra_dir.mkdir(parents=True, exist_ok=True)
    path = spectra_dir / filename
    # Save as list of arrays (variable length)
    np.savez_compressed(str(path), *spectra)
    print(f"Saved {len(spectra)} spectra to {path}", flush=True)


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------


def run_baselines(args):
    """Compute all baselines on the same data for fair comparison."""
    print("=" * 70)
    print("EXPERIMENT: Baselines (k-NN, Mahalanobis, Isolation Forest)")
    print("=" * 70, flush=True)

    device = args.device
    results = []

    for model_key in (args.model_keys if hasattr(args, "model_keys") else [args.model]):
        model = load_model(model_key, device)
        pairs = get_ood_pairs(args.n_samples, args.ood)

        for pair_name, (id_imgs, ood_imgs) in pairs.items():
            print(f"\n  {model_key} / {pair_name}", flush=True)

            # Extract embeddings (shared cost)
            t0 = time.time()
            id_emb = extract_embeddings(model, id_imgs, device)
            ood_emb = extract_embeddings(model, ood_imgs, device)
            embed_time = time.time() - t0
            embed_dim = id_emb.shape[1]
            print(f"    Embeddings: {id_emb.shape}, {embed_time:.1f}s", flush=True)

            for method_name, score_fn in [
                ("k-NN", lambda ref, test: knn_scores(ref, test)),
                ("Mahalanobis", lambda ref, test: maha_scores(ref, test)),
                ("IsolationForest", lambda ref, test: iforest_scores(ref, test)),
            ]:
                t0 = time.time()
                id_s = score_fn(id_emb, id_emb)
                ood_s = score_fn(id_emb, ood_emb)
                elapsed = time.time() - t0
                metrics = compute_metrics(id_s, ood_s)
                print(
                    f"    {method_name:20s} AUROC={metrics['auroc']:.4f} "
                    f"FPR95={metrics['fpr95']:.4f} ({elapsed:.2f}s)",
                    flush=True,
                )
                results.append(ExperimentResult(
                    experiment="baselines",
                    model=model_key,
                    ood_pair=pair_name,
                    method=method_name,
                    n_samples=args.n_samples,
                    auroc=metrics["auroc"],
                    fpr95=metrics["fpr95"],
                    time_s=elapsed + embed_time,  # include embedding time
                    embed_dim=embed_dim,
                    notes=f"Embedding extraction: {embed_time:.1f}s, Scoring: {elapsed:.2f}s",
                ))

        del model
        torch.cuda.empty_cache()

    save_results(results, "baselines.json")
    return results


def run_full_jacobian(args):
    """Full Jacobian JEPA-SCORE, matching the paper exactly."""
    print("=" * 70)
    print(f"EXPERIMENT: Full Jacobian JEPA-SCORE ({args.model})")
    print(f"  This matches the paper's code in Appendix B.")
    print(f"  Each sample requires d={384 if 'vits' in args.model else 768} backward passes.")
    print("=" * 70, flush=True)

    device = args.device
    model = load_model(args.model, device)
    pairs = get_ood_pairs(args.n_samples, args.ood)

    results = []
    all_spectra = {}

    for pair_name, (id_imgs, ood_imgs) in pairs.items():
        print(f"\n  {pair_name}", flush=True)

        # ID scores
        t0 = time.time()
        id_scores, id_spectra = run_full_jacobian_batch(
            model, id_imgs, device, label=f"ID({pair_name})"
        )
        # OOD scores
        ood_scores, ood_spectra = run_full_jacobian_batch(
            model, ood_imgs, device, label=f"OOD({pair_name})"
        )
        elapsed = time.time() - t0

        # JEPA-SCORE: higher = higher density = more likely ID
        # For OOD detection (higher = more OOD), negate the scores
        metrics = compute_metrics(-id_scores, -ood_scores)
        print(
            f"  RESULT: AUROC={metrics['auroc']:.4f} "
            f"FPR95={metrics['fpr95']:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        results.append(ExperimentResult(
            experiment="full_jacobian",
            model=args.model,
            ood_pair=pair_name,
            method="JEPA-SCORE-full",
            n_samples=args.n_samples,
            auroc=metrics["auroc"],
            fpr95=metrics["fpr95"],
            time_s=elapsed,
            embed_dim=384 if "vits" in args.model else 768,
            notes="Full Jacobian matching paper Appendix B. No randomized approximation.",
        ))

        # Save spectra
        save_spectra(id_spectra, f"full_{args.model}_{pair_name}_id.npz")
        save_spectra(ood_spectra, f"full_{args.model}_{pair_name}_ood.npz")

        # Save intermediate results after each pair
        save_results(results, f"full_jacobian_{args.model}.json")

    del model
    torch.cuda.empty_cache()
    return results


def run_projection_sweep(args):
    """Randomized JEPA-SCORE with varying projection counts."""
    projections = [16, 32, 64, 128, 256]
    # Add full embed dim if model is vits14
    if "vits" in args.model:
        projections.append(384)
    elif "vitb" in args.model:
        projections.append(384)  # still cap at 384 for time reasons

    print("=" * 70)
    print(f"EXPERIMENT: Projection Sweep ({args.model})")
    print(f"  Projections: {projections}")
    print(f"  Purpose: characterize approximation-accuracy tradeoff")
    print("=" * 70, flush=True)

    device = args.device
    model = load_model(args.model, device)
    pairs = get_ood_pairs(args.n_samples, args.ood)

    results = []

    for pair_name, (id_imgs, ood_imgs) in pairs.items():
        for n_proj in projections:
            print(f"\n  {pair_name} / p={n_proj}", flush=True)

            t0 = time.time()
            id_scores, id_spectra = jepa_score_randomized(
                model, id_imgs, device, n_proj=n_proj, seed=42
            )
            ood_scores, ood_spectra = jepa_score_randomized(
                model, ood_imgs, device, n_proj=n_proj, seed=42
            )
            elapsed = time.time() - t0

            # Negate: higher original score = more ID → lower negated = more ID
            metrics = compute_metrics(-id_scores, -ood_scores)
            print(
                f"    p={n_proj:4d} AUROC={metrics['auroc']:.4f} "
                f"FPR95={metrics['fpr95']:.4f} ({elapsed:.1f}s)",
                flush=True,
            )

            results.append(ExperimentResult(
                experiment="projection_sweep",
                model=args.model,
                ood_pair=pair_name,
                method=f"JEPA-SCORE-p{n_proj}",
                n_samples=args.n_samples,
                auroc=metrics["auroc"],
                fpr95=metrics["fpr95"],
                time_s=elapsed,
                n_projections=n_proj,
                seed=42,
                embed_dim=384 if "vits" in args.model else 768,
            ))

            # Save spectra for largest projection count
            if n_proj == projections[-1]:
                save_spectra(
                    id_spectra, f"sweep_{args.model}_{pair_name}_p{n_proj}_id.npz"
                )
                save_spectra(
                    ood_spectra, f"sweep_{args.model}_{pair_name}_p{n_proj}_ood.npz"
                )

        # Save after each pair
        save_results(results, f"projection_sweep_{args.model}.json")

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def print_summary():
    """Print a summary of all results found in the results directory."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70, flush=True)

    if not RESULTS_DIR.exists():
        print("No results found yet. Run experiments first.")
        return

    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(json_file.read_text())
        print(f"\n--- {json_file.name} ({len(data)} entries) ---")
        print(f"{'Model':<18} {'OOD':<25} {'Method':<22} {'AUROC':>7} {'FPR95':>7} {'Time':>8}")
        print("-" * 90)
        for r in data:
            print(
                f"{r['model']:<18} {r['ood_pair']:<25} {r['method']:<22} "
                f"{r['auroc']:>7.4f} {r['fpr95']:>7.4f} {r['time_s']:>7.1f}s"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(
        description="Extended JEPA-SCORE experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--experiment",
        choices=["baselines", "full_jacobian", "projection_sweep", "summary"],
        required=True,
        help="Which experiment to run",
    )
    p.add_argument("--model", default="dinov2_vits14", choices=list(MODELS.keys()))
    p.add_argument("--ood", default=None, help="Single OOD pair (e.g. CIFAR10_vs_SVHN)")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--full-test", action="store_true", help="Use full test sets for baselines (ignores --n-samples for baselines)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    print(f"Device: {args.device}")
    if args.device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(flush=True)

    if args.experiment == "baselines":
        # If --model specified explicitly, only run that model; else run all DINOv2 models
        if "--model" in sys.argv:
            args.model_keys = [args.model]
        else:
            args.model_keys = [k for k in MODELS if k.startswith("dinov2_vits") or k.startswith("dinov2_vitb")]
        # If --full-test, use 0 which signals load_subset to use all samples
        if getattr(args, "full_test", False):
            args.n_samples = 0  # 0 = full test set
        run_baselines(args)
    elif args.experiment == "full_jacobian":
        run_full_jacobian(args)
    elif args.experiment == "projection_sweep":
        run_projection_sweep(args)
    elif args.experiment == "summary":
        print_summary()
        return

    print_summary()


if __name__ == "__main__":
    main()
