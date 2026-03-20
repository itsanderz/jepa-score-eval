# JEPA-SCORE Evaluation

**First independent quantitative evaluation of JEPA-SCORE for out-of-distribution detection on pretrained vision encoders.**

Paper: *"Understanding Jacobian-Based Density Estimation for OOD Detection on Pretrained Vision Encoders"*
Author: [Andrew Morgan](https://x.com/itsanderz)

## Key Findings

| Encoder | Method | SVHN | CIFAR-100 | DTD |
|---------|--------|------|-----------|-----|
| DINOv2 ViT-S | Mahalanobis | **1.000** | **1.000** | **1.000** |
| DINOv2 ViT-S | k-NN | 0.965 | 0.915 | 1.000 |
| DINOv2 ViT-S | **JEPA-SCORE** | 0.841 | 0.638 | 0.995 |
| DINOv2 ViT-L | **JEPA-SCORE** | **0.477** | 0.566 | 0.810 |
| MetaCLIP | **JEPA-SCORE** | **0.020** | 0.527 | 0.978 |

- JEPA-SCORE substantially **underperforms** distance-based baselines
- **Inverse scaling**: AUROC degrades as encoder size increases (0.841 → 0.673 → 0.477)
- **MetaCLIP inversion**: contrastive encoders produce nearly perfectly inverted scores (0.020)
- **Tail-weighting**: dropping top 20% of Jacobian singular values improves AUROC by +0.023
- Full spectral analysis reveals dominant singular values are anti-discriminative (correlation > 0.997)

## Repository Contents

```
├── jepa_score.py              # JEPA-SCORE implementation (full + randomized)
├── run_extended.py            # Full experiment runner
├── verify_tail_weighting.py   # Verification script for tail-weighting claims
├── cross_validate_tail.py     # 5-fold cross-validation of tail-weighting
├── generate_figures.py        # Generate all paper figures
├── zero_cost_analyses.py      # Score correlation + ensemble analysis
├── full_results_gpu.json      # All experimental results (raw)
├── all_results_combined.json  # Combined results across all runs
├── results_A_spectra/         # Saved SVD spectra: DINOv2 ViT-S (n=500)
├── results_B_spectra/         # Saved SVD spectra: DINOv2 ViT-B
├── results_C_spectra/         # Saved SVD spectra: DINOv2 ViT-L (n=200)
├── results_D_spectra/         # Saved SVD spectra: MetaCLIP (n=200)
├── requirements.txt           # Python dependencies
├── REPRODUCIBILITY.md         # Detailed reproduction guide
└── LICENSE                    # MIT License
```

## Quick Start

```bash
git clone https://github.com/itsanderz/jepa-score-eval.git
cd jepa-score-eval
pip install -r requirements.txt

# Compute JEPA-SCORE for a single image
python -c "
import torch
from jepa_score import jepa_score_full

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()
x = torch.randn(3, 224, 224).cuda()
score, singular_values = jepa_score_full(model, x)
print(f'JEPA-SCORE: {score:.2f}')
print(f'Top-5 singular values: {singular_values[:5]}')
"
```

## Reproduce Paper Results

```bash
# Table 1: Full Jacobian JEPA-SCORE (ViT-S, n=500)
python run_extended.py --experiment full_jacobian --model dinov2_vits14 --n-samples 500

# Table 2: Projection sweep (p = 16 to 384)
python run_extended.py --experiment projection_sweep --model dinov2_vits14 --n-samples 200

# Verify tail-weighting claims
python verify_tail_weighting.py

# 5-fold cross-validation
python cross_validate_tail.py

# Generate all figures
python generate_figures.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for complete step-by-step instructions including expected outputs, compute requirements, and known issues.

## Datasets

All datasets download automatically via `torchvision`:
- **CIFAR-10** (in-distribution) — MIT license
- **SVHN** (far-OOD) — research use
- **CIFAR-100** (near-OOD) — MIT license
- **DTD** (texture-OOD) — research use

## Models

All models from Meta's official `torch.hub`:
- **DINOv2 ViT-S/14** (21M params) — Apache 2.0
- **DINOv2 ViT-B/14** (86M params) — Apache 2.0
- **DINOv2 ViT-L/14** (304M params) — Apache 2.0
- **MetaCLIP ViT-B/16** (86M params) — CC-BY-NC 4.0 (research use)

## Implementation Note

Our JEPA-SCORE implementation matches the original paper's Appendix B pseudocode exactly. No official code was released by the authors. See `jepa_score.py` for a side-by-side comparison with the original pseudocode.

## Citation

```bibtex
@article{morgan2026jepa_eval,
  title={Understanding Jacobian-Based Density Estimation for OOD Detection
         on Pretrained Vision Encoders},
  author={Morgan, Andrew},
  journal={arXiv preprint},
  year={2026}
}
```

## Original Paper

This work evaluates JEPA-SCORE from:

```bibtex
@article{balestriero2025gaussian,
  title={Gaussian Embeddings: How JEPAs Secretly Learn Your Data Density},
  author={Balestriero, Randall and Ballas, Nicolas and Rabbat, Michael and LeCun, Yann},
  journal={arXiv preprint arXiv:2510.05949},
  year={2025}
}
```

## Acknowledgments

- Adam Hibble for introducing the world models research direction
- Experiments aided by Claude (Anthropic) for code development and literature review
- All experimental results, analysis, and conclusions are the author's own
- Compute provided by [Vast.ai](https://cloud.vast.ai/?ref_id=104180) (~$15 total)

## Contact

- Twitter/X: [@itsanderz](https://x.com/itsanderz)
- Email: andrew@weareabsurd.co

## License

MIT — use freely, cite if you publish.
