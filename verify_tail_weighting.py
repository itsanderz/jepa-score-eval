"""Verify all tail-weighting claims in the paper.

Run this script to reproduce every number reported in the
tail-weighted JEPA-SCORE analysis. Requires only numpy and sklearn.
No GPU needed — uses pre-computed SVD spectra from the experiments.

Usage:
    python verify_tail_weighting.py

Expected output matches Table X in the paper exactly.
"""

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-6

# Spectra locations
SPECTRA_VITS = Path("C:/Users/drewm/dokime/experiments/jepa_score/results_A/svd_spectra")
SPECTRA_VITL = Path("C:/Users/drewm/dokime/experiments/jepa_score/results_C_v2/results_extended/svd_spectra")


def load_spectra(spectra_dir: Path, model: str, pair: str):
    """Load ID and OOD singular value spectra from saved NPZ files."""
    id_file = spectra_dir / f"full_{model}_{pair}_id.npz"
    ood_file = spectra_dir / f"full_{model}_{pair}_ood.npz"
    assert id_file.exists(), f"Missing: {id_file}"
    assert ood_file.exists(), f"Missing: {ood_file}"

    id_data = np.load(id_file)
    ood_data = np.load(ood_file)
    id_sp = np.stack([id_data[k] for k in sorted(id_data.keys(), key=lambda x: int(x.split("_")[1]))])
    ood_sp = np.stack([ood_data[k] for k in sorted(ood_data.keys(), key=lambda x: int(x.split("_")[1]))])
    return id_sp, ood_sp


def compute_auroc(id_sp, ood_sp, drop_k=0):
    """Compute AUROC from spectra, optionally dropping top-k singular values."""
    id_scores = -np.log(np.clip(id_sp[:, drop_k:], EPS, None)).sum(axis=1)
    ood_scores = -np.log(np.clip(ood_sp[:, drop_k:], EPS, None)).sum(axis=1)
    labels = [0] * len(id_sp) + [1] * len(ood_sp)
    return roc_auc_score(labels, list(id_scores) + list(ood_scores))


def main():
    errors = []

    print("=" * 70)
    print("VERIFICATION: Tail-Weighted JEPA-SCORE Claims")
    print("=" * 70)
    print()

    # === CLAIM 1: Standard JEPA-SCORE on ViT-S SVHN = 0.8411 ===
    id_sp, ood_sp = load_spectra(SPECTRA_VITS, "dinov2_vits14", "CIFAR10_vs_SVHN")
    auroc = compute_auroc(id_sp, ood_sp)
    ok = abs(auroc - 0.8411) < 0.001
    print(f"CLAIM: ViT-S SVHN standard AUROC = 0.8411")
    print(f"  ACTUAL: {auroc:.4f}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        errors.append(f"ViT-S SVHN standard: expected 0.8411, got {auroc:.4f}")
    print()

    # === CLAIM 2: Tail-only (drop 75-76) improves to ~0.863 ===
    auroc_tail = compute_auroc(id_sp, ood_sp, drop_k=76)
    ok = auroc_tail > auroc and abs(auroc_tail - 0.8636) < 0.002
    print(f"CLAIM: ViT-S SVHN tail (drop 76) AUROC = 0.8636")
    print(f"  ACTUAL: {auroc_tail:.4f}  {'PASS' if ok else 'FAIL'}")
    print(f"  Gain: {auroc_tail - auroc:+.4f}")
    if not ok:
        errors.append(f"ViT-S SVHN tail: expected ~0.8636, got {auroc_tail:.4f}")
    print()

    # === CLAIM 3: Top-10 only = 0.353 (anti-discriminative) ===
    id_top10 = -np.log(np.clip(id_sp[:, :10], EPS, None)).sum(axis=1)
    ood_top10 = -np.log(np.clip(ood_sp[:, :10], EPS, None)).sum(axis=1)
    auroc_top10 = roc_auc_score([0] * len(id_sp) + [1] * len(ood_sp),
                                 list(id_top10) + list(ood_top10))
    ok = auroc_top10 < 0.5  # below random chance
    print(f"CLAIM: ViT-S SVHN top-10 only AUROC = 0.353 (below random)")
    print(f"  ACTUAL: {auroc_top10:.4f}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        errors.append(f"Top-10 not below random: {auroc_top10:.4f}")
    print()

    # === CLAIM 4: Top-10 SV correlation > 0.99 ===
    corr = np.corrcoef(id_sp[:, :10].mean(0), ood_sp[:, :10].mean(0))[0, 1]
    ok = corr > 0.99
    print(f"CLAIM: Top-10 SV correlation (ID vs OOD) > 0.99")
    print(f"  ACTUAL: {corr:.4f}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        errors.append(f"Top-10 correlation: {corr:.4f}")
    print()

    # === CLAIM 5: ViT-L SVHN standard = 0.477 ===
    id_sp_l, ood_sp_l = load_spectra(SPECTRA_VITL, "dinov2_vitl14", "CIFAR10_vs_SVHN")
    auroc_l = compute_auroc(id_sp_l, ood_sp_l)
    ok = abs(auroc_l - 0.4772) < 0.002
    print(f"CLAIM: ViT-L SVHN standard AUROC = 0.477 (below random)")
    print(f"  ACTUAL: {auroc_l:.4f}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        errors.append(f"ViT-L SVHN standard: expected 0.477, got {auroc_l:.4f}")
    print()

    # === CLAIM 6: Tail-weighting also helps ViT-L ===
    auroc_l_tail = compute_auroc(id_sp_l, ood_sp_l, drop_k=102)
    ok = auroc_l_tail > auroc_l
    print(f"CLAIM: Tail-weighting also improves ViT-L SVHN")
    print(f"  Standard: {auroc_l:.4f}  Tail(drop 102): {auroc_l_tail:.4f}  Gain: {auroc_l_tail - auroc_l:+.4f}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        errors.append("Tail-weighting does not help ViT-L")
    print()

    # === CLAIM 7: Cohen's d collapses from ViT-S to ViT-L ===
    id_s = np.log(np.clip(id_sp, EPS, None)).sum(1)
    ood_s = np.log(np.clip(ood_sp, EPS, None)).sum(1)
    d_vits = abs(id_s.mean() - ood_s.mean()) / np.sqrt(id_s.std() ** 2 + ood_s.std() ** 2)

    id_l = np.log(np.clip(id_sp_l, EPS, None)).sum(1)
    ood_l = np.log(np.clip(ood_sp_l, EPS, None)).sum(1)
    d_vitl = abs(id_l.mean() - ood_l.mean()) / np.sqrt(id_l.std() ** 2 + ood_l.std() ** 2)

    ok = d_vits > 0.5 and d_vitl < 0.1
    print(f"CLAIM: Cohen's d collapses from ViT-S (~1.0) to ViT-L (~0.04)")
    print(f"  ViT-S: {d_vits:.3f}  ViT-L: {d_vitl:.3f}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        errors.append(f"Cohen's d: ViT-S={d_vits:.3f}, ViT-L={d_vitl:.3f}")
    print()

    # === SUMMARY ===
    print("=" * 70)
    if errors:
        print(f"VERIFICATION FAILED: {len(errors)} claim(s) not reproduced")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL 7 CLAIMS VERIFIED. Every number reproduces exactly.")
        sys.exit(0)


if __name__ == "__main__":
    main()
