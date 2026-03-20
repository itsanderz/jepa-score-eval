"""Zero-cost analyses on existing JEPA-SCORE experiment data.

Runs entirely on CPU using previously saved results and SVD spectra.
No GPU required. No new experiments.

Analyses:
  1. Score correlation (JEPA-SCORE vs Mahalanobis per sample)
  2. Epsilon sensitivity (vary clipping threshold)
  3. Per-class AUROC (CIFAR-10's 10 classes)
  4. Ensemble test (combine JEPA-SCORE + Mahalanobis)

Usage:
  python zero_cost_analyses.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

# Paths
RESULTS_DIR = Path("C:/Users/drewm/dokime/experiments/jepa_score")
RESULTS_A = RESULTS_DIR / "results_A"
SPECTRA_A = RESULTS_A / "svd_spectra"
PRELIM = RESULTS_DIR / "full_results_gpu.json"
OUT_DIR = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

EPS_DEFAULT = 1e-6


# ── Analysis 1: Epsilon Sensitivity ──────────────────────────────────────────

def analysis_epsilon_sensitivity():
    """Vary eps and recompute JEPA-SCORE AUROC from saved SVD spectra."""
    print("=== Epsilon Sensitivity ===")

    epsilons = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    pairs = ["CIFAR10_vs_SVHN", "CIFAR10_vs_CIFAR100", "CIFAR10_vs_Textures"]
    pair_labels = {"CIFAR10_vs_SVHN": "SVHN", "CIFAR10_vs_CIFAR100": "CIFAR-100", "CIFAR10_vs_Textures": "DTD"}
    colors = {"CIFAR10_vs_SVHN": "#e63946", "CIFAR10_vs_CIFAR100": "#457b9d", "CIFAR10_vs_Textures": "#2a9d8f"}

    results = {}

    for pair in pairs:
        id_file = SPECTRA_A / f"full_dinov2_vits14_{pair}_id.npz"
        ood_file = SPECTRA_A / f"full_dinov2_vits14_{pair}_ood.npz"

        if not id_file.exists() or not ood_file.exists():
            print(f"  Skipping {pair}: spectra not found")
            continue

        id_data = np.load(id_file)
        ood_data = np.load(ood_file)

        id_spectra = [id_data[k] for k in sorted(id_data.keys(), key=lambda x: int(x.split("_")[1]))]
        ood_spectra = [ood_data[k] for k in sorted(ood_data.keys(), key=lambda x: int(x.split("_")[1]))]

        aurocs = []
        for eps in epsilons:
            # Recompute JEPA-SCORE with this epsilon
            id_scores = [np.log(np.clip(sv, max(eps, 1e-30), None)).sum() for sv in id_spectra]
            ood_scores = [np.log(np.clip(sv, max(eps, 1e-30), None)).sum() for sv in ood_spectra]

            # AUROC: higher JEPA-SCORE = more in-distribution, so negate for OOD detection
            labels = [0] * len(id_scores) + [1] * len(ood_scores)
            scores = [-s for s in id_scores] + [-s for s in ood_scores]
            auroc = roc_auc_score(labels, scores)
            aurocs.append(auroc)

        results[pair] = aurocs
        print(f"  {pair_labels[pair]}: AUROC range [{min(aurocs):.4f}, {max(aurocs):.4f}]")

    # Plot
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    eps_labels = ["0"] + [f"1e-{int(-np.log10(e))}" for e in epsilons[1:]]

    for pair in pairs:
        if pair in results:
            ax.plot(range(len(epsilons)), results[pair], "o-", color=colors[pair],
                    label=f"vs {pair_labels[pair]}", markersize=4, linewidth=1.5)

    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels(eps_labels, rotation=45)
    ax.set_xlabel(r"Clipping threshold $\varepsilon$")
    ax.set_ylabel("AUROC")
    ax.legend()
    ax.grid(True, alpha=0.2)
    ax.set_title("JEPA-SCORE sensitivity to $\\varepsilon$ (ViT-S/14)")

    fig.savefig(OUT_DIR / "fig_epsilon_sensitivity.pdf")
    fig.savefig(OUT_DIR / "fig_epsilon_sensitivity.png")
    print("  Saved fig_epsilon_sensitivity.pdf/png")
    plt.close(fig)

    return results


# ── Analysis 2: Score Correlation ────────────────────────────────────────────

def analysis_score_correlation():
    """Compute per-sample correlation between JEPA-SCORE and Mahalanobis."""
    print("\n=== Score Correlation ===")
    print("  Note: This requires per-sample scores. Using SVD spectra to recompute JEPA-SCORE")
    print("  and would need per-sample Mahalanobis scores (not saved in current format).")
    print("  For now, computing summary statistics from aggregate results.")
    print()

    # Load aggregate results
    with open(RESULTS_A / "baselines.json") as f:
        baselines = json.load(f)
    with open(RESULTS_A / "full_jacobian_dinov2_vits14.json") as f:
        full_jac = json.load(f)

    print("  Aggregate comparison (ViT-S/14):")
    print(f"  {'Pair':<25} {'Mahalanobis':>12} {'JEPA-full':>12} {'k-NN':>12}")
    print("  " + "-" * 65)
    for fj in full_jac:
        pair = fj["ood_pair"]
        jepas = fj["auroc"]
        maha = next((b["auroc"] for b in baselines if b["ood_pair"] == pair and b["method"] == "Mahalanobis" and b["model"] == "dinov2_vits14"), None)
        knn = next((b["auroc"] for b in baselines if b["ood_pair"] == pair and b["method"] == "k-NN" and b["model"] == "dinov2_vits14"), None)
        print(f"  {pair:<25} {maha:>12.4f} {jepas:>12.4f} {knn:>12.4f}")

    # For a proper per-sample correlation, we would need to save per-sample
    # Mahalanobis scores during the experiment. Flag this for the Vast.ai run.
    print()
    print("  ACTION: Next Vast.ai run should save per-sample scores for correlation analysis.")
    print("  Adding this to the run_extended.py modification list.")


# ── Analysis 3: Epsilon + Spectrum Stats ─────────────────────────────────────

def analysis_spectrum_stats():
    """Compute summary statistics of SVD spectra."""
    print("\n=== SVD Spectrum Statistics ===")

    pairs = ["CIFAR10_vs_SVHN", "CIFAR10_vs_CIFAR100", "CIFAR10_vs_Textures"]
    pair_labels = {"CIFAR10_vs_SVHN": "SVHN", "CIFAR10_vs_CIFAR100": "CIFAR-100", "CIFAR10_vs_Textures": "DTD"}

    for pair in pairs:
        id_file = SPECTRA_A / f"full_dinov2_vits14_{pair}_id.npz"
        ood_file = SPECTRA_A / f"full_dinov2_vits14_{pair}_ood.npz"

        if not id_file.exists():
            continue

        id_data = np.load(id_file)
        ood_data = np.load(ood_file)

        id_spectra = np.stack([id_data[k] for k in sorted(id_data.keys(), key=lambda x: int(x.split("_")[1]))])
        ood_spectra = np.stack([ood_data[k] for k in sorted(ood_data.keys(), key=lambda x: int(x.split("_")[1]))])

        # Compute scores
        id_scores = np.log(np.clip(id_spectra, EPS_DEFAULT, None)).sum(axis=1)
        ood_scores = np.log(np.clip(ood_spectra, EPS_DEFAULT, None)).sum(axis=1)

        # Stats
        id_mean, id_std = id_scores.mean(), id_scores.std()
        ood_mean, ood_std = ood_scores.mean(), ood_scores.std()
        separation = abs(id_mean - ood_mean) / np.sqrt(id_std**2 + ood_std**2)

        # Fraction of singular values < eps
        id_near_zero = (id_spectra < EPS_DEFAULT).mean()
        ood_near_zero = (ood_spectra < EPS_DEFAULT).mean()

        # Top-k singular value overlap
        id_top10 = id_spectra[:, :10].mean(axis=0)
        ood_top10 = ood_spectra[:, :10].mean(axis=0)
        top10_corr = np.corrcoef(id_top10, ood_top10)[0, 1]

        print(f"\n  {pair_labels[pair]}:")
        print(f"    ID score:  {id_mean:.2f} ± {id_std:.2f}")
        print(f"    OOD score: {ood_mean:.2f} ± {ood_std:.2f}")
        print(f"    Separation (Cohen's d): {separation:.3f}")
        print(f"    SV near zero (<eps): ID={id_near_zero:.4f}, OOD={ood_near_zero:.4f}")
        print(f"    Top-10 SV correlation (ID vs OOD means): {top10_corr:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Zero-cost analyses on existing JEPA-SCORE data")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print()

    eps_results = analysis_epsilon_sensitivity()
    analysis_score_correlation()
    analysis_spectrum_stats()

    print("\n\nDone! Check output figures in:", OUT_DIR)
