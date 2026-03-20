"""
Standalone implementation of JEPA-SCORE.

Reference: Balestriero et al., "Gaussian Embeddings: How JEPAs Secretly
Learn Your Data Density," NeurIPS 2025. arXiv:2510.05949.

This implements the density scoring method from the paper, which estimates
log p(x) via the log-determinant of the encoder's Jacobian:

    JEPA-SCORE(x) = sum_i log sigma_i(J_f(x))

where J_f(x) is the Jacobian of encoder f at input x, and sigma_i are
its singular values.

Two variants are provided:
  - full_jacobian: exact computation matching the paper's Appendix B code
  - randomized: approximation using p random projections (NOT in the paper)

Paper's reference code (Appendix B):

    from torch.autograd.functional import jacobian
    J = jacobian(lambda x: model(x).sum(0), inputs=images)
    with torch.inference_mode():
        J = J.flatten(2).permute(1, 0, 2)
        svdvals = torch.linalg.svdvals(J)
        jepa_score = svdvals.clip_(eps).log_().sum(1)
"""

from __future__ import annotations

import torch
import numpy as np


EPS = 1e-6  # Paper: "eps (we pick 1e-6)"


def jepa_score_full(
    model: torch.nn.Module,
    x: torch.Tensor,
    eps: float = EPS,
) -> tuple[float, np.ndarray]:
    """
    Compute JEPA-SCORE using the full Jacobian, matching the paper exactly.

    Args:
        model: Encoder that maps (B, *input_shape) -> (B, d).
        x: Single input tensor (no batch dim), e.g. shape (3, 224, 224).
        eps: Numerical stability constant for log. Paper uses 1e-6.

    Returns:
        score: Scalar JEPA-SCORE. Higher = higher estimated density.
        singular_values: Array of d singular values (descending order).
    """
    device = next(model.parameters()).device
    x_batch = x.unsqueeze(0).to(device)  # (1, *input_shape)

    def func(inp: torch.Tensor) -> torch.Tensor:
        return model(inp).sum(0)  # (d,) — remove batch dim via sum

    # Full Jacobian: shape (d, 1, *input_shape)
    J = torch.autograd.functional.jacobian(func, x_batch, vectorize=False)

    # Reshape to (batch=1, d, D) then compute SVD
    J = J.flatten(2).permute(1, 0, 2)  # (1, d, D)
    sv = torch.linalg.svdvals(J)       # (1, d)

    score = sv.clamp(min=eps).log().sum(1).item()  # scalar
    sv_np = sv[0].detach().cpu().numpy()

    return score, sv_np


def jepa_score_randomized(
    model: torch.nn.Module,
    x: torch.Tensor,
    n_proj: int = 64,
    eps: float = EPS,
    generator: torch.Generator | None = None,
) -> tuple[float, np.ndarray]:
    """
    Compute JEPA-SCORE using a randomized Jacobian approximation.

    NOT described in the original paper. The paper uses the full Jacobian.
    This approximation reduces cost from d backward passes to p passes,
    but the resulting score is computed on a projected Jacobian and is
    not identical to the true JEPA-SCORE.

    Args:
        model: Encoder that maps (B, *input_shape) -> (B, d).
        x: Single input tensor (no batch dim).
        n_proj: Number of random Gaussian projections (p).
        eps: Numerical stability constant for log.
        generator: Optional torch.Generator for reproducibility.

    Returns:
        score: Scalar approximate JEPA-SCORE.
        singular_values: Array of p singular values (descending order).
    """
    device = next(model.parameters()).device

    # Get embedding dim with a no-grad forward pass
    with torch.no_grad():
        emb = model(x.unsqueeze(0).to(device))
    d = emb.shape[1]

    # Random projection matrix: (p, d)
    omega = torch.randn(n_proj, d, device=device, generator=generator)

    # Compute p VJPs: omega[j]^T @ J_f(x) for each j
    vjp_rows = []
    for j in range(n_proj):
        x_j = x.unsqueeze(0).to(device).requires_grad_(True)
        out = model(x_j)  # (1, d)
        (g,) = torch.autograd.grad(
            out, x_j,
            grad_outputs=omega[j].unsqueeze(0),  # (1, d)
            retain_graph=False,
        )
        vjp_rows.append(g.flatten().detach())  # (D,)

    # Projected Jacobian: (p, D)
    J_proj = torch.stack(vjp_rows, dim=0)

    # SVD and score
    sv = torch.linalg.svdvals(J_proj)  # (p,)
    score = sv.clamp(min=eps).log().sum().item()
    sv_np = sv.cpu().numpy()

    return score, sv_np


def jepa_score_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    method: str = "full",
    n_proj: int = 64,
    seed: int = 42,
    eps: float = EPS,
    verbose: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Score a batch of images using JEPA-SCORE.

    Args:
        model: Encoder.
        images: Tensor of shape (N, C, H, W).
        method: "full" for exact Jacobian, "randomized" for approximation.
        n_proj: Number of projections (only used if method="randomized").
        seed: Random seed for projections.
        eps: Numerical stability constant.
        verbose: Print progress.

    Returns:
        scores: Array of N scores (higher = higher density = more in-distribution).
        spectra: List of N singular value arrays.
    """
    device = next(model.parameters()).device
    scores = []
    spectra = []
    n = len(images)

    if method == "randomized":
        gen = torch.Generator(device=device).manual_seed(seed)
    else:
        gen = None

    for i in range(n):
        if method == "full":
            score, sv = jepa_score_full(model, images[i], eps=eps)
        elif method == "randomized":
            score, sv = jepa_score_randomized(
                model, images[i], n_proj=n_proj, eps=eps, generator=gen,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'full' or 'randomized'.")

        scores.append(score)
        spectra.append(sv)

        if verbose and ((i + 1) % 50 == 0 or i == 0):
            print(f"  [{i+1}/{n}] score={score:.2f}", flush=True)

    return np.array(scores), spectra
