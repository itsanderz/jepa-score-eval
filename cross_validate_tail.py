"""Cross-validated tail-weighted JEPA-SCORE.

Properly validates the tail-weighting improvement using 5-fold CV.
Eliminates overfitting concern: optimal drop_k found on training folds,
evaluated on held-out fold.

Also tests generalization across encoder sizes:
- Find drop_k on ViT-S, test on ViT-L (cross-encoder transfer)

No GPU needed. Uses pre-computed SVD spectra.
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from pathlib import Path

EPS = 1e-6
SPECTRA_VITS = Path("C:/Users/drewm/dokime/experiments/jepa_score/results_A/svd_spectra")
SPECTRA_VITL = Path("C:/Users/drewm/dokime/experiments/jepa_score/results_C_v2/results_extended/svd_spectra")


def load_spectra(spectra_dir, model, pair):
    id_data = np.load(spectra_dir / f"full_{model}_{pair}_id.npz")
    ood_data = np.load(spectra_dir / f"full_{model}_{pair}_ood.npz")
    id_sp = np.stack([id_data[k] for k in sorted(id_data.keys(), key=lambda x: int(x.split("_")[1]))])
    ood_sp = np.stack([ood_data[k] for k in sorted(ood_data.keys(), key=lambda x: int(x.split("_")[1]))])
    return id_sp, ood_sp


def auroc_with_drop(id_sp, ood_sp, drop_k=0):
    id_scores = -np.log(np.clip(id_sp[:, drop_k:], EPS, None)).sum(axis=1)
    ood_scores = -np.log(np.clip(ood_sp[:, drop_k:], EPS, None)).sum(axis=1)
    labels = [0] * len(id_sp) + [1] * len(ood_sp)
    return roc_auc_score(labels, list(id_scores) + list(ood_scores))


def find_best_drop_k(id_sp, ood_sp, d):
    """Grid search for best drop_k on given data."""
    best_auroc, best_k = auroc_with_drop(id_sp, ood_sp, 0), 0
    for frac in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        k = int(d * frac)
        auroc = auroc_with_drop(id_sp, ood_sp, k)
        if auroc > best_auroc:
            best_auroc, best_k = auroc, k
    return best_k, best_auroc


def main():
    print("=" * 70)
    print("CROSS-VALIDATED TAIL-WEIGHTED JEPA-SCORE")
    print("=" * 70)
    print()

    for pair_name, pair in [("SVHN", "CIFAR10_vs_SVHN"), ("CIFAR-100", "CIFAR10_vs_CIFAR100")]:
        id_sp, ood_sp = load_spectra(SPECTRA_VITS, "dinov2_vits14", pair)
        n_id, d = id_sp.shape
        n_ood = ood_sp.shape[0]

        print(f"=== ViT-S vs {pair_name} (n_id={n_id}, n_ood={n_ood}, d={d}) ===")
        print()

        # Standard (no CV needed — deterministic)
        auroc_std = auroc_with_drop(id_sp, ood_sp, 0)
        print(f"  Standard JEPA-SCORE: {auroc_std:.4f}")

        # --- 5-Fold Cross-Validation ---
        # We split BOTH ID and OOD into folds, find best drop_k on train, eval on test
        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        cv_aurocs_std = []
        cv_aurocs_tail = []
        cv_drop_ks = []

        id_indices = np.arange(n_id)
        ood_indices = np.arange(n_ood)

        for fold, ((id_train_idx, id_test_idx), (ood_train_idx, ood_test_idx)) in enumerate(
            zip(kf.split(id_indices), kf.split(ood_indices))
        ):
            # Train fold: find optimal drop_k
            id_train = id_sp[id_train_idx]
            ood_train = ood_sp[ood_train_idx]
            best_k, _ = find_best_drop_k(id_train, ood_train, d)
            cv_drop_ks.append(best_k)

            # Test fold: evaluate with that drop_k
            id_test = id_sp[id_test_idx]
            ood_test = ood_sp[ood_test_idx]

            auroc_test_std = auroc_with_drop(id_test, ood_test, 0)
            auroc_test_tail = auroc_with_drop(id_test, ood_test, best_k)

            cv_aurocs_std.append(auroc_test_std)
            cv_aurocs_tail.append(auroc_test_tail)

            print(f"  Fold {fold+1}: drop_k={best_k:3d}  standard={auroc_test_std:.4f}  tail={auroc_test_tail:.4f}  gain={auroc_test_tail-auroc_test_std:+.4f}")

        mean_std = np.mean(cv_aurocs_std)
        mean_tail = np.mean(cv_aurocs_tail)
        std_std = np.std(cv_aurocs_std)
        std_tail = np.std(cv_aurocs_tail)

        print()
        print(f"  5-Fold CV Standard:     {mean_std:.4f} +/- {std_std:.4f}")
        print(f"  5-Fold CV Tail-weighted: {mean_tail:.4f} +/- {std_tail:.4f}")
        print(f"  CV Gain:                {mean_tail - mean_std:+.4f}")
        print(f"  Selected drop_ks:       {cv_drop_ks}")
        print(f"  Improvement consistent: {all(t > s for t, s in zip(cv_aurocs_tail, cv_aurocs_std))}")
        print()

    # --- Cross-Encoder Transfer ---
    print("=" * 70)
    print("CROSS-ENCODER TRANSFER: Find drop_k on ViT-S, test on ViT-L")
    print("=" * 70)
    print()

    for pair_name, pair in [("SVHN", "CIFAR10_vs_SVHN"), ("CIFAR-100", "CIFAR10_vs_CIFAR100")]:
        id_s, ood_s = load_spectra(SPECTRA_VITS, "dinov2_vits14", pair)
        id_l, ood_l = load_spectra(SPECTRA_VITL, "dinov2_vitl14", pair)

        d_s = id_s.shape[1]
        d_l = id_l.shape[1]

        # Find best fraction on ViT-S
        best_k_s, _ = find_best_drop_k(id_s, ood_s, d_s)
        best_frac = best_k_s / d_s

        # Apply same fraction to ViT-L
        drop_k_l = int(d_l * best_frac)

        auroc_l_std = auroc_with_drop(id_l, ood_l, 0)
        auroc_l_tail = auroc_with_drop(id_l, ood_l, drop_k_l)

        print(f"  {pair_name}:")
        print(f"    ViT-S best: drop {best_k_s}/{d_s} = {best_frac*100:.0f}%")
        print(f"    Applied to ViT-L: drop {drop_k_l}/{d_l}")
        print(f"    ViT-L standard: {auroc_l_std:.4f}")
        print(f"    ViT-L tail:     {auroc_l_tail:.4f}  gain: {auroc_l_tail-auroc_l_std:+.4f}")
        print()


if __name__ == "__main__":
    main()
