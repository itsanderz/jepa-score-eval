# Negative Results: What We Tested and What Didn't Work

This document records experiments beyond Paper 1 that produced negative or null results. We share these to save the community time and compute.

All experiments use the same codebase and methodology as Paper 1. Raw data is available in this repository and in `experiments/` directories.

---

## 1. SIGReg Enables JEPA-SCORE but Still Loses to Baselines

**Hypothesis:** Adding SIGReg (covariance → Identity regularization from LeJEPA, arXiv:2511.08544) to non-JEPA encoders would make JEPA-SCORE competitive with distance-based baselines.

**Setup:** ResNet-18 trained on CIFAR-10 with BCS (Barlow Twins variant), evaluated on CIFAR-10 vs SVHN.

**Results:**

| Training | AUROC | Direction |
|----------|-------|-----------|
| BCS baseline | 0.033 | Inverted |
| BCS + Jacobian spectral reg | 0.056 | Inverted |
| VICReg | 0.051 | Inverted |
| BCS + SIGReg (from scratch) | 0.514 | Correct |
| BCS + SIGReg (fine-tune 50ep) | 0.630 | Correct |
| DINOv2 ViT-L + SIGReg (fine-tune) | 0.521 | Correct |

**Verdict:** SIGReg successfully flips JEPA-SCORE from inverted to correct. However, the best AUROC (0.630) is still 33 points below k-NN (0.965) on the same embeddings. SIGReg addresses the theoretical requirement (isotropic Gaussian embeddings) but does not make JEPA-SCORE practically competitive.

**Takeaway:** The isotropic Gaussian assumption in JEPA-SCORE theory (Balestriero et al. 2025) is necessary but not sufficient for competitive OOD detection. Even when embeddings are approximately Gaussian, the Jacobian log-determinant captures less OOD-relevant information than simple embedding distances.

**Compute cost:** ~$4 on Vast.ai (RTX 4090).

---

## 2. Jacobian Spectral Features Don't Beat Embedding k-NN

**Hypothesis:** The scalar JEPA-SCORE collapses 384+ dimensions of Jacobian singular values into one number. Using the full singular value vector as features for k-NN might recover OOD information the scalar destroys.

**Setup:** k-NN on full SVD spectra (384-1024 dimensions) vs scalar JEPA-SCORE vs standard embedding k-NN. Tested on DINOv2 ViT-S/B/L and MetaCLIP.

**Results (CIFAR-10 vs SVHN):**

| Model | Scalar JEPA-SCORE | Best Spectral k-NN | Embedding k-NN |
|-------|-------------------|--------------------|--------------  |
| DINOv2 ViT-S | 0.834 | 0.870 | **0.965** |
| DINOv2 ViT-B | 0.667 | 0.975 | **0.990** |
| DINOv2 ViT-L | 0.488 | 0.914 | **1.000** |
| MetaCLIP | 0.012 | 0.990 | **1.000** |

**Verdict:** Spectral features recover substantial OOD signal that the scalar score destroys (e.g., MetaCLIP: 0.012 → 0.990). However, **embedding k-NN beats spectral k-NN on every single model**, while being 100-1000x cheaper to compute (one forward pass vs hundreds of backward passes).

**Takeaway:** The Jacobian contains OOD information, but it's a subset of what the embeddings already capture. There is no scenario where computing the expensive Jacobian is preferable to using the embeddings directly.

**Additional finding:** For MetaCLIP, spectral k-NN (0.990) ≈ negated scalar score (0.988). The apparent improvement is just implicit score negation, not new information from the full spectrum.

**Compute cost:** $0 (used existing saved spectra from Paper 1).

---

## 3. JEPA-SCORE Data Curation Doesn't Beat Random Selection

**Hypothesis:** JEPA-SCORE could rank training images by estimated density, and training on the highest-density subset would improve downstream classifier performance — the use case suggested (but not tested) in the original paper.

**Setup:** DINOv2 ViT-S JEPA-SCORE on 5000 CIFAR-10 training images. Trained logistic regression classifiers on subsets selected by different curation methods. Evaluated on 10000 CIFAR-10 test images.

**Curation methods tested:**
- Random selection (3 seeds, averaged)
- Top-k by embedding k-NN density ("typical" samples)
- Bottom-k by embedding density ("hard" samples)
- Top-k by JEPA-SCORE ("high density")
- Bottom-k by JEPA-SCORE ("low density")
- Disagreement region (Jacobian atypical but embedding typical)
- Combined (normalized JEPA + embedding density)

**Results (downstream accuracy):**

| Fraction | Random | Emb Typical | Emb Hard | JEPA Typical | JEPA Hard | Disagreement | Combined |
|----------|--------|-------------|----------|--------------|-----------|-------------|----------|
| 10% | **0.919** | ~0.91 | ~0.91 | ~0.91 | ~0.91 | ~0.91 | ~0.91 |
| 30% | **0.934** | ~0.93 | ~0.93 | ~0.93 | ~0.93 | ~0.93 | ~0.93 |
| 50% | 0.938 | ~0.93 | ~0.93 | **0.938** | ~0.93 | ~0.93 | ~0.93 |
| 70% | 0.939 | ~0.93 | **0.939** | ~0.93 | ~0.93 | ~0.93 | ~0.93 |

**Key finding:** JEPA-SCORE and embedding density are weakly anti-correlated (Pearson r = -0.29), confirming they measure different things. However, **no curation method consistently beats random selection.** DINOv2 features are so powerful that even a random 10% subset achieves 91.9% accuracy.

**Takeaway:** On clean, well-curated datasets (CIFAR-10), no density-based curation signal — whether from embeddings or Jacobians — improves over random selection. The original paper's suggestion of using JEPA-SCORE for data curation remains untested on messy/noisy datasets where curation would actually matter.

**Compute cost:** ~$2 on Vast.ai (RTX 4090, ~3.5 hours for 5000 Jacobian computations).

---

## 4. Jacobian Doesn't Predict World Model Prediction Reliability

**Hypothesis:** The Jacobian spectral properties of a world model's predictor (DINO-WM, arXiv:2411.04983) at each timestep predict whether the next-step prediction will be accurate — providing a training-free reliability signal without ensembles.

**Setup:** Pretrained DINO-WM on point_maze (authors' checkpoint). Computed randomized Jacobian (p=32 projections) of the ViT predictor at each timestep of 200 test steps across 10 trajectories. Correlated spectral metrics with L2 prediction error.

**Results:**

| Metric | Spearman r | p-value | Signal? |
|--------|-----------|---------|---------|
| log_det | +0.233 | 0.0009 | Statistically significant but weak |
| spectral_concentration | -0.083 | 0.24 | Not significant |
| effective_rank | +0.083 | 0.24 | Not significant |
| condition_number | -0.065 | 0.36 | Not significant |
| frobenius_norm | +0.023 | 0.74 | Nothing |

**Verdict:** The best predictor (log_det) achieves only Spearman r=0.233. While statistically significant (p<0.001), this is far too weak to be useful as a reliability signal. A practitioner would need r>0.5 for the signal to be actionable.

**Takeaway:** The Jacobian spectral properties of a learned world model predictor do not meaningfully indicate when predictions will fail. The local geometric sensitivity captured by the Jacobian is not aligned with prediction error in this setting. Ensemble disagreement remains the standard approach for world model uncertainty estimation.

**Compute cost:** ~$0.40 on Vast.ai (RTX 4090, ~1 hour including data download).

---

## Summary

| Experiment | Result | Practical value of Jacobian |
|-----------|--------|---------------------------|
| OOD detection (Paper 1) | JEPA-SCORE < k-NN by 12-36 pts | None |
| SIGReg fix | Flips direction but still < k-NN by 33 pts | None |
| Spectral features | < embedding k-NN on every model | None |
| Data curation | No method beats random | None |
| World model reliability | r=0.233 (too weak) | None |

**The consistent finding across all experiments:** The Jacobian of pretrained vision encoders and world model predictors does not provide actionable information beyond what simpler, cheaper methods already capture. The theoretical connection between anti-collapse regularization and density estimation (Balestriero et al. 2025) is valid but does not translate to practical advantages in any downstream task we tested.

We share these results so others don't spend compute exploring these dead ends.

---

## Reproducibility

All code is in this repository. Raw results for experiments 2-3 are in the `experiments/` directory of the [Dokime repo](https://github.com/dokime-ai/dokime). The world model pilot used the pretrained DINO-WM checkpoint from [OSF](https://osf.io/bmw48/).

## Contact

Andrew Morgan — [@itsanderz](https://x.com/itsanderz) — andrew@weareabsurd.co
