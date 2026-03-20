# JEPA-SCORE Evaluation

**First independent quantitative evaluation of JEPA-SCORE for out-of-distribution detection on pretrained vision encoders.**

Paper: "Understanding Jacobian-Based Density Estimation for OOD Detection on Pretrained Vision Encoders"
Author: Andrew Morgan

## Key Findings

- JEPA-SCORE substantially underperforms distance-based baselines (Mahalanobis: 1.000 AUROC vs JEPA-SCORE: 0.841 on ViT-S SVHN)
- Inverse scaling: JEPA-SCORE **degrades** as encoder size increases (ViT-S: 0.841 → ViT-B: 0.673 → ViT-L: 0.477)
- MetaCLIP (contrastive encoder) produces nearly perfectly inverted JEPA-SCORE (0.020 AUROC)
- Dominant Jacobian singular values are anti-discriminative; dropping top 20% improves AUROC by +0.023

## Files

- `jepa_score.py` — Standalone JEPA-SCORE implementation (full Jacobian + randomized approximation)
- `run_extended.py` — Full experiment runner (multiple encoders, OOD pairs, baselines)
- `full_results_gpu.json` — All experimental results
- `all_results_combined.json` — Combined results across all runs
- `results_A_spectra/` — Saved SVD spectra for ViT-S

## Quick Start

```bash
pip install torch torchvision scikit-learn scipy numpy

# Compute JEPA-SCORE for a single image
python -c "
import torch
from jepa_score import jepa_score_full

model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').cuda().eval()
x = torch.randn(3, 224, 224).cuda()
score, singular_values = jepa_score_full(model, x)
print(f'JEPA-SCORE: {score:.2f}')
"

# Run full evaluation
python run_extended.py --experiment full_jacobian --model dinov2_vits14 --n-samples 500
```

## Reference

If you use this code, please cite:

```bibtex
@article{morgan2026jepa_eval,
  title={Understanding Jacobian-Based Density Estimation for OOD Detection on Pretrained Vision Encoders},
  author={Morgan, Andrew},
  journal={arXiv preprint},
  year={2026}
}
```

## Acknowledgments

Adam Hibble for introducing the world models research direction. Experiments aided by Claude (Anthropic). All results and conclusions are the author's own.

## License

MIT
