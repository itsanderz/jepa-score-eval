# Reproducibility Guide

Every number in the paper can be reproduced from this repository. This document explains exactly how.

## Environment

Our experiments were run on:
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) on Vast.ai
- **CUDA**: 12.6–13.0
- **PyTorch**: 2.6.0–2.10.0
- **Python**: 3.10–3.14
- **OS**: Ubuntu 22.04 (Vast.ai instances)

Results are deterministic given the same seed, but may vary slightly (~0.01 AUROC) across different PyTorch versions due to numerical differences in autograd and SVD implementations.

## Datasets

All datasets are downloaded automatically via `torchvision.datasets`:

| Dataset | Role | Source | Size | License |
|---------|------|--------|------|---------|
| CIFAR-10 | In-distribution | `torchvision.datasets.CIFAR10` | 60K images (32×32) | MIT |
| SVHN | Far-OOD | `torchvision.datasets.SVHN` | 99K test images (32×32) | Non-commercial research |
| CIFAR-100 | Near-OOD | `torchvision.datasets.CIFAR100` | 60K images (32×32) | MIT |
| DTD | Texture-OOD | `torchvision.datasets.DTD` | 5,640 images (variable) | Custom (research) |

All images are resized to 224×224 and normalized with ImageNet statistics:
- Mean: (0.485, 0.456, 0.406)
- Std: (0.229, 0.224, 0.225)

## Pretrained Models

All models are loaded via `torch.hub` from Meta's official repositories:

| Model | Source | Params | Embed Dim | License |
|-------|--------|--------|-----------|---------|
| DINOv2 ViT-S/14 | `facebookresearch/dinov2` | 21M | 384 | Apache 2.0 |
| DINOv2 ViT-B/14 | `facebookresearch/dinov2` | 86M | 768 | Apache 2.0 |
| DINOv2 ViT-L/14 | `facebookresearch/dinov2` | 304M | 1024 | Apache 2.0 |
| MetaCLIP ViT-B/16 | via `open_clip` | 86M | 512 | CC-BY-NC 4.0 |

MetaCLIP is used under research fair use, consistent with the original JEPA-SCORE paper's own evaluation.

## JEPA-SCORE Implementation

Our implementation in `jepa_score.py` matches the original paper's Appendix B pseudocode exactly:

**Original (Balestriero et al. 2025, Appendix B):**
```python
from torch.autograd.functional import jacobian
J = jacobian(lambda x: model(x).sum(0), inputs=images)
with torch.inference_mode():
    J = J.flatten(2).permute(1, 0, 2)
    svdvals = torch.linalg.svdvals(J)
    jepa_score = svdvals.clip_(eps).log_().sum(1)
```

**Ours (`jepa_score.py`, `jepa_score_full` function):**
```python
J = torch.autograd.functional.jacobian(func, x_batch, vectorize=False)
J = J.flatten(2).permute(1, 0, 2)
sv = torch.linalg.svdvals(J)
score = sv.clamp(min=eps).log().sum(1).item()
```

The only differences:
1. We process one sample at a time (GPU memory management)
2. We return singular values for spectral analysis
3. We use `vectorize=False` (identical results; `vectorize=True` is 2x faster with negligible numerical difference <0.4 on scores of magnitude ~800)
4. Epsilon = 1e-6 (matching the paper)

## Reproducing Key Results

### Table 1: Main AUROC Results

**ViT-S/14, CIFAR-10 vs SVHN (n=500):**
```bash
python run_extended.py --experiment full_jacobian --model dinov2_vits14 --ood CIFAR10_vs_SVHN --n-samples 500
```
Expected: JEPA-SCORE AUROC ≈ 0.841, Mahalanobis ≈ 1.000, k-NN ≈ 0.965

**ViT-S/14, CIFAR-10 vs CIFAR-100 (n=500):**
```bash
python run_extended.py --experiment full_jacobian --model dinov2_vits14 --ood CIFAR10_vs_CIFAR100 --n-samples 500
```
Expected: JEPA-SCORE AUROC ≈ 0.638, Mahalanobis ≈ 1.000

**ViT-B/14, CIFAR-10 vs SVHN (n=500):**
```bash
python run_extended.py --experiment full_jacobian --model dinov2_vitb14 --ood CIFAR10_vs_SVHN --n-samples 500
```
Expected: JEPA-SCORE AUROC ≈ 0.673

**ViT-L/14, all OOD pairs (n=200):**
```bash
python run_extended.py --experiment full_jacobian --model dinov2_vitl14 --n-samples 200
```
Expected: SVHN ≈ 0.477, CIFAR-100 ≈ 0.566, DTD ≈ 0.810

**MetaCLIP, all OOD pairs (n=200):**
```bash
python run_extended.py --experiment full_jacobian --model metaclip_b16 --n-samples 200
```
Expected: SVHN ≈ 0.020 (inverted), CIFAR-100 ≈ 0.527, DTD ≈ 0.978

### Table 2: Projection Sweep

```bash
python run_extended.py --experiment projection_sweep --model dinov2_vits14 --n-samples 200
```
This runs p ∈ {16, 32, 64, 128, 256, 384} on all three OOD pairs.

### Tail-Weighting Analysis

The tail-weighting results use saved SVD spectra in `results_A_spectra/`. To reproduce:

```python
import numpy as np
from sklearn.metrics import roc_auc_score

# Load spectra
id_data = np.load('results_A_spectra/full_dinov2_vits14_CIFAR10_vs_SVHN_id.npz')
ood_data = np.load('results_A_spectra/full_dinov2_vits14_CIFAR10_vs_SVHN_ood.npz')

id_spec = np.stack([id_data[f'arr_{i}'] for i in range(len(id_data.files))])
ood_spec = np.stack([ood_data[f'arr_{i}'] for i in range(len(ood_data.files))])

# Full JEPA-SCORE
full_id = np.sum(np.log(np.clip(id_spec, 1e-6, None)), axis=1)
full_ood = np.sum(np.log(np.clip(ood_spec, 1e-6, None)), axis=1)
labels = np.concatenate([np.ones(len(full_id)), np.zeros(len(full_ood))])
print(f"Full: AUROC = {roc_auc_score(labels, np.concatenate([full_id, full_ood])):.4f}")

# Tail-only (drop top 20%)
k = int(384 * 0.2)  # = 76
tail_id = np.sum(np.log(np.clip(id_spec[:, k:], 1e-6, None)), axis=1)
tail_ood = np.sum(np.log(np.clip(ood_spec[:, k:], 1e-6, None)), axis=1)
print(f"Tail (drop {k}): AUROC = {roc_auc_score(labels, np.concatenate([tail_id, tail_ood])):.4f}")
```
Expected: Full ≈ 0.841, Tail ≈ 0.864

## Random Seeds

All experiments use seed 42 for:
- Dataset subsampling (`np.random.RandomState(42)`)
- Random projection matrices for randomized JEPA-SCORE
- Isolation Forest random state

Multi-seed experiments (Table 2) additionally use seeds 123 and 456.

## Compute Requirements

| Experiment | GPU | Time per 500 samples | VRAM |
|-----------|-----|---------------------|------|
| ViT-S full Jacobian | RTX 4090 | ~35 min | ~2.5 GB |
| ViT-B full Jacobian | RTX 4090 | ~55 min | ~4 GB |
| ViT-L full Jacobian | RTX 4090 | ~90 min | ~8 GB |
| Baselines (all) | Any GPU | <30 sec | <1 GB |

Total compute for all experiments: ~$15 on Vast.ai (RTX 4090 at ~$0.30/hr).

## Known Issues

1. **Embedding dimension metadata bug**: Our logging code hardcodes `embed_dim=768` for non-ViT-S models. The actual Jacobian computation uses the true model dimensions and is unaffected. This is a logging-only bug.

2. **ViT-L Mahalanobis baseline**: Returns AUROC ≈ 0.0 because the covariance matrix is singular when n=200 < d=1024. This is a known limitation of Mahalanobis distance at high dimensions with small samples, not a bug.

3. **Sample size variation**: ViT-S/B experiments use n=500; ViT-L/MetaCLIP use n=200 due to computational cost. ViT-S at n=5000 yields AUROC=0.806 (vs 0.841 at n=500), indicating variance at small n.

## Contact

Andrew Morgan — andrew@weareabsurd.co
