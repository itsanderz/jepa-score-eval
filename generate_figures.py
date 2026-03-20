"""Generate publication-quality figures for the JEPA-SCORE paper.

Figures:
  1. Projection sweep: AUROC vs number of projections (3 OOD pairs)
  2. SVD spectrum: mean singular value distribution for ID vs OOD

Usage:
  python generate_figures.py

Requires: matplotlib, numpy
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paths
RESULTS_DIR = Path("C:/Users/drewm/dokime/experiments/jepa_score")
RESULTS_A = RESULTS_DIR / "results_A"
RESULTS_B = RESULTS_DIR / "results_B"
RESULTS_C = RESULTS_DIR / "results_C"
OUT_DIR = Path(__file__).parent

# Publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})


# ── Figure 1: Projection Sweep ─────────────────────────────────────────────

def fig_projection_sweep():
    """AUROC vs number of projections for each OOD pair."""
    with open(RESULTS_B / "projection_sweep_dinov2_vits14.json") as f:
        data = json.load(f)

    # Also load the full Jacobian results for reference lines
    with open(RESULTS_A / "full_jacobian_dinov2_vits14.json") as f:
        full_data = json.load(f)

    # Also load baselines for context
    with open(RESULTS_A / "baselines.json") as f:
        baselines = json.load(f)

    # Organize sweep data by OOD pair
    pairs = {}
    for r in data:
        pair = r["ood_pair"]
        if pair not in pairs:
            pairs[pair] = {"p": [], "auroc": []}
        pairs[pair]["p"].append(r["n_projections"])
        pairs[pair]["auroc"].append(r["auroc"])

    # Full Jacobian reference points
    full_ref = {r["ood_pair"]: r["auroc"] for r in full_data}

    # Baseline reference (k-NN, ViT-S only)
    knn_ref = {}
    for r in baselines:
        if r["model"] == "dinov2_vits14" and r["method"] == "k-NN":
            knn_ref[r["ood_pair"]] = r["auroc"]

    # Nice display names
    pair_names = {
        "CIFAR10_vs_SVHN": "SVHN",
        "CIFAR10_vs_CIFAR100": "CIFAR-100",
        "CIFAR10_vs_Textures": "DTD",
    }

    colors = {
        "CIFAR10_vs_SVHN": "#e63946",
        "CIFAR10_vs_CIFAR100": "#457b9d",
        "CIFAR10_vs_Textures": "#2a9d8f",
    }

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    for pair_key in ["CIFAR10_vs_SVHN", "CIFAR10_vs_CIFAR100", "CIFAR10_vs_Textures"]:
        d = pairs[pair_key]
        name = pair_names[pair_key]
        color = colors[pair_key]

        # Sweep line
        ax.plot(d["p"], d["auroc"], "o-", color=color, label=f"JEPA-SCORE vs {name}",
                markersize=4, linewidth=1.5)

        # Full Jacobian reference (diamond marker)
        if pair_key in full_ref:
            # Place at x = 400 (slightly past 384) to distinguish from sweep
            ax.plot(400, full_ref[pair_key], "D", color=color, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.5)

        # k-NN baseline (dashed horizontal line)
        if pair_key in knn_ref:
            ax.axhline(y=knn_ref[pair_key], color=color, linestyle="--",
                       alpha=0.4, linewidth=1)

    # Annotations
    ax.annotate("k-NN baselines", xy=(300, 0.97), fontsize=8, color="gray", style="italic")
    ax.annotate("◆ = full Jacobian", xy=(300, 0.55), fontsize=8, color="gray", style="italic")

    ax.set_xlabel("Number of random projections $p$")
    ax.set_ylabel("AUROC")
    ax.set_xlim(0, 420)
    ax.set_ylim(0.5, 1.02)
    ax.set_xticks([16, 32, 64, 128, 256, 384])
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2)

    fig.savefig(OUT_DIR / "fig_projection_sweep.pdf")
    fig.savefig(OUT_DIR / "fig_projection_sweep.png")
    print("Saved fig_projection_sweep.pdf/png")
    plt.close(fig)


# ── Figure 2: SVD Spectrum ──────────────────────────────────────────────────

def fig_svd_spectrum():
    """Mean singular value spectrum for ID vs OOD samples."""
    spectra_dir_a = RESULTS_A / "svd_spectra"
    spectra_dir_c = RESULTS_C / "svd_spectra"

    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

    pairs_to_plot = [
        ("CIFAR10_vs_SVHN", "dinov2_vits14", "SVHN (ViT-S)", spectra_dir_a),
        ("CIFAR10_vs_CIFAR100", "dinov2_vits14", "CIFAR-100 (ViT-S)", spectra_dir_a),
        ("CIFAR10_vs_SVHN", "dinov2_vitb14", "SVHN (ViT-B)", spectra_dir_c),
    ]

    for ax, (pair, model, title, spec_dir) in zip(axes, pairs_to_plot):
        # Try to load spectra
        id_file = spec_dir / f"full_{model}_{pair}_id.npz"
        ood_file = spec_dir / f"full_{model}_{pair}_ood.npz"

        if not id_file.exists() or not ood_file.exists():
            ax.text(0.5, 0.5, "Spectra not\navailable", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
            ax.set_title(title)
            continue

        id_data = np.load(id_file)
        ood_data = np.load(ood_file)

        # Each sample is stored as a separate array: arr_0, arr_1, ...
        # Stack them into (n_samples, n_singular_values)
        id_spectra = np.stack([id_data[k] for k in sorted(id_data.keys(), key=lambda x: int(x.split("_")[1]))])
        ood_spectra = np.stack([ood_data[k] for k in sorted(ood_data.keys(), key=lambda x: int(x.split("_")[1]))])

        # Compute mean and std of log singular values
        id_log_mean = np.mean(np.log10(np.clip(id_spectra, 1e-10, None)), axis=0)
        ood_log_mean = np.mean(np.log10(np.clip(ood_spectra, 1e-10, None)), axis=0)
        id_log_std = np.std(np.log10(np.clip(id_spectra, 1e-10, None)), axis=0)
        ood_log_std = np.std(np.log10(np.clip(ood_spectra, 1e-10, None)), axis=0)

        indices = np.arange(len(id_log_mean))

        ax.plot(indices, id_log_mean, color="#457b9d", linewidth=1.2, label="ID (CIFAR-10)")
        ax.fill_between(indices, id_log_mean - id_log_std, id_log_mean + id_log_std,
                         color="#457b9d", alpha=0.15)

        ax.plot(indices, ood_log_mean, color="#e63946", linewidth=1.2, label="OOD")
        ax.fill_between(indices, ood_log_mean - ood_log_std, ood_log_mean + ood_log_std,
                         color="#e63946", alpha=0.15)

        ax.set_title(title)
        ax.set_xlabel("Singular value index")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("log₁₀(singular value)")
    axes[0].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_svd_spectrum.pdf")
    fig.savefig(OUT_DIR / "fig_svd_spectrum.png")
    print("Saved fig_svd_spectrum.pdf/png")
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating figures...")
    print(f"Results dir: {RESULTS_DIR}")
    print(f"Output dir: {OUT_DIR}")
    print()

    fig_projection_sweep()
    fig_svd_spectrum()

    print("\nDone! Figures saved to:", OUT_DIR)
