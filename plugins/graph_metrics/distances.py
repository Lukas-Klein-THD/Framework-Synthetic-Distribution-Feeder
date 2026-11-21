"""
Distances: two sets of descriptor vectors -> named scores.

Each declares its direction, so a scorecard mixing "lower is better" and
"higher is better" stays readable.
"""

from __future__ import annotations

import numpy as np

from gridgen.evaluation.base import Direction, Distance

_BASE_SIGMAS = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :]
                      - 2.0 * a @ b.T, 0.0)


class MaximumMeanDiscrepancy(Distance):
    """RBF MMD^2, maximised over bandwidths scaled by the median heuristic."""

    @property
    def name(self) -> str:
        return "mmd"

    @property
    def direction(self) -> Direction:
        return Direction.LOWER_IS_BETTER

    def __call__(self, real, generated):
        d_rr, d_gg, d_rg = _sqdist(real, real), _sqdist(generated, generated), \
            _sqdist(real, generated)
        median = np.median(d_rg)
        scale = np.sqrt(median) if median > 0 else 1.0
        best = 0.0
        for sigma in _BASE_SIGMAS * scale:
            gamma = 1.0 / (2.0 * sigma * sigma)
            value = (np.exp(-gamma * d_rr).mean() + np.exp(-gamma * d_gg).mean()
                     - 2.0 * np.exp(-gamma * d_rg).mean())
            best = max(best, float(value))
        return {"value": best}


class FrechetDistance(Distance):
    """Frechet distance between Gaussians fitted to the two embeddings."""

    @property
    def name(self) -> str:
        return "fid"

    @property
    def direction(self) -> Direction:
        return Direction.LOWER_IS_BETTER

    def __call__(self, real, generated):
        mu_r, mu_g = real.mean(0), generated.mean(0)
        cov_r, cov_g = np.cov(real, rowvar=False), np.cov(generated, rowvar=False)
        eig = np.linalg.eigvals(cov_r @ cov_g)
        trace = np.sqrt(np.clip(eig.real, 0.0, None)).sum()
        diff = mu_r - mu_g
        return {"value": float(diff @ diff + np.trace(cov_r) + np.trace(cov_g)
                               - 2.0 * trace)}


class PrecisionRecallDensityCoverage(Distance):
    """Manifold precision/recall/density/coverage (nearest-neighbour based)."""

    def __init__(self, nearest_k: int = 5):
        self.k = nearest_k

    @property
    def name(self) -> str:
        return "prdc"

    @property
    def direction(self) -> Direction:
        return Direction.HIGHER_IS_BETTER

    def __call__(self, real, generated):
        radius_r = self._radius(real)
        radius_g = self._radius(generated)
        d = np.sqrt(_sqdist(real, generated))
        inside = d <= radius_r[:, None]
        return {
            "precision": float(inside.any(0).mean()),
            "recall": float((d <= radius_g[None, :]).any(1).mean()),
            "density": float((1.0 / self.k) * inside.sum(0).mean()),
            "coverage": float(inside.any(1).mean()),
        }

    def _radius(self, x: np.ndarray) -> np.ndarray:
        d = np.sqrt(_sqdist(x, x))
        d.sort(axis=1)
        return d[:, min(self.k, d.shape[1] - 1)]
