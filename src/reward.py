"""Synthetic reward function with known ground-truth ATE.

R = beta_w * W + beta_typo * has_typo + eps, an additive decomposition that
satisfies RATE's Assumption 2 by construction, so experiments built on this
reward isolate violations of Assumption 1 (direction-dependent rewrite
errors) without also confounding through Assumption 2.
"""

import numpy as np

DEFAULT_BETA_W = 1.0
DEFAULT_BETA_TYPO = -0.5
DEFAULT_NOISE_SD = 1.0


def reward(
    w: np.ndarray,
    has_typo: np.ndarray,
    rng: np.random.Generator,
    beta_w: float = DEFAULT_BETA_W,
    beta_typo: float = DEFAULT_BETA_TYPO,
    noise_sd: float = DEFAULT_NOISE_SD,
) -> np.ndarray:
    """R = beta_w * w + beta_typo * has_typo + eps, eps ~ N(0, noise_sd^2)."""
    w = np.asarray(w, dtype=float)
    has_typo = np.asarray(has_typo, dtype=float)
    eps = rng.normal(0.0, noise_sd, size=len(w))
    return beta_w * w + beta_typo * has_typo + eps


def true_ate(beta_w: float = DEFAULT_BETA_W) -> float:
    """Closed-form ATE = ATT = ATU = beta_w (homogeneous effect)."""
    return beta_w
