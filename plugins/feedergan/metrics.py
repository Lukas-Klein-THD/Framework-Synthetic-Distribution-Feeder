"""
Structural and attributed metrics.

Descriptors turn feeders into vectors; distances compare two sets of vectors.
Keeping them apart means a new descriptor works with every distance and vice
versa — adding an embedding does not require reimplementing MMD.

Three points of interpretation are built in:

* Embedding scores have **no absolute meaning**; read them against the
  real-versus-real reference the evaluator computes.
* **Precision and recall must be read together.**  A model producing a narrow
  set of plausible feeders scores high precision and near-zero recall.
* MMD uses the **maximum over a bandwidth range** rather than a single kernel
  width, since one bandwidth can hide a difference the next one exposes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gridgen.core.feeder import Feeder
from gridgen.core.registry import register
from gridgen.evaluation.base import Descriptor, DescriptorMetric, Direction, Distance

_BASE_SIGMAS = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])


def _undirected(adjacency: np.ndarray) -> np.ndarray:
    a = (np.asarray(adjacency) != 0)
    u = (a | a.T).astype(float)
    np.fill_diagonal(u, 0.0)
    return u


# ── descriptors ──────────────────────────────────────────────────────────────
@register("metric", "degree-histogram")
class DegreeHistogram(Descriptor):
    """Normalised degree distribution — local connectivity."""

    def __init__(self, max_degree: int = 24):
        self.max_degree = max_degree

    @property
    def name(self) -> str:
        return "degree"

    def __call__(self, feeders: Sequence[Feeder]) -> np.ndarray:
        rows = []
        for f in feeders:
            deg = _undirected(f.adjacency).sum(1).astype(int)
            hist = np.bincount(np.clip(deg, 0, self.max_degree),
                               minlength=self.max_degree + 1).astype(float)
            rows.append(hist / max(hist.sum(), 1.0))
        return np.stack(rows)


class LaplacianSpectrum(Descriptor):
    """Histogram of Laplacian eigenvalues — global structure."""

    def __init__(self, bins: int = 200):
        self.bins = bins

    @property
    def name(self) -> str:
        return "spectrum"

    def __call__(self, feeders: Sequence[Feeder]) -> np.ndarray:
        rows = []
        for f in feeders:
            u = _undirected(f.adjacency)
            laplacian = np.diag(u.sum(1)) - u
            eigenvalues = np.linalg.eigvalsh(laplacian)
            hist, _ = np.histogram(eigenvalues, bins=self.bins, range=(0.0, 2.0))
            rows.append(hist.astype(float) / max(hist.sum(), 1.0))
        return np.stack(rows)


class RandomGINEmbedding(Descriptor):
    """Graph embedding from an untrained GIN.

    An untrained encoder is a fixed, dataset-agnostic feature map, so the metric
    cannot overfit to either population.  Node inputs may carry attributes, in
    which case the embedding reflects the *joint* of structure and attributes
    rather than structure alone.
    """

    def __init__(self, layers: int = 3, hidden: int = 35, seed: int = 0,
                 encoder=None):
        self.layers, self.hidden, self.seed = layers, hidden, seed
        self.encoder = encoder                     # optional attribute encoder
        self._weights: list[tuple[np.ndarray, np.ndarray]] | None = None

    @property
    def name(self) -> str:
        return "gin-attributed" if self.encoder else "gin"

    def _init(self, in_dim: int) -> None:
        rng = np.random.default_rng(self.seed)
        dims = [in_dim] + [self.hidden] * self.layers
        self._weights = [
            (rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1])),
             rng.normal(0, 0.1, dims[i + 1]))
            for i in range(self.layers)]

    def __call__(self, feeders: Sequence[Feeder]) -> np.ndarray:
        rows = []
        for f in feeders:
            u = _undirected(f.adjacency)
            x = (self.encoder(f) if self.encoder
                 else np.ones((f.num_nodes, 1), dtype=float))
            if self._weights is None:
                self._init(x.shape[1])
            pooled = []
            h = x
            for weight, bias in self._weights:
                h = np.maximum((h + u @ h) @ weight + bias, 0.0)   # sum aggregation
                pooled.append(h.sum(0))                            # sum readout
            rows.append(np.concatenate(pooled))                    # jumping knowledge
        return np.stack(rows)


# ── distances ────────────────────────────────────────────────────────────────
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


# ── ready-made metric suite ──────────────────────────────────────────────────
def structural_metrics(max_degree: int = 24, seed: int = 0) -> list[DescriptorMetric]:
    """Topology-only metrics."""
    gin = RandomGINEmbedding(seed=seed)
    return [
        DescriptorMetric(DegreeHistogram(max_degree), MaximumMeanDiscrepancy()),
        DescriptorMetric(LaplacianSpectrum(), MaximumMeanDiscrepancy()),
        DescriptorMetric(gin, FrechetDistance()),
        DescriptorMetric(gin, PrecisionRecallDensityCoverage()),
    ]


def attributed_metrics(encoder, seed: int = 0) -> list[DescriptorMetric]:
    """Metrics over the joint of structure and attributes."""
    gin = RandomGINEmbedding(seed=seed, encoder=encoder)
    return [
        DescriptorMetric(gin, FrechetDistance()),
        DescriptorMetric(gin, PrecisionRecallDensityCoverage()),
    ]
