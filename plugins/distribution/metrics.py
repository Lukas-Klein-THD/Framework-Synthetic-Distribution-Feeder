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
    """Histogram of normalised-Laplacian eigenvalues — global structure.

    The normalised Laplacian ``I - D^-1/2 A D^-1/2`` has its spectrum in [0, 2]
    for every graph, so a fixed histogram over that range loses nothing.
    """

    def __init__(self, bins: int = 200):
        self.bins = bins

    @property
    def name(self) -> str:
        return "spectrum"

    def __call__(self, feeders: Sequence[Feeder]) -> np.ndarray:
        edges = np.linspace(0.0, 2.0, self.bins + 1)
        rows = []
        for f in feeders:
            u = _undirected(f.adjacency)
            d = u.sum(1)
            inv = np.divide(1.0, np.sqrt(d), out=np.zeros_like(d), where=d > 0)
            laplacian = np.eye(len(d)) - inv[:, None] * u * inv[None, :]
            hist, _ = np.histogram(np.linalg.eigvalsh(laplacian), bins=edges)
            rows.append(hist.astype(float) / max(hist.sum(), 1.0))
        return np.stack(rows)


class RandomGINEmbedding(Descriptor):
    """Graph embedding from an untrained GIN (Thompson et al., 2022).

    An untrained encoder is a fixed, dataset-agnostic feature map, so the metric
    cannot overfit to either population.  Configuration of GGM-metrics: a linear
    projection, three GIN layers ``h <- MLP(h + A h)`` with two-layer MLPs and
    hidden width 35, sum pooling of every layer, concatenated (jumping
    knowledge).  Node input is the node degree, or the attribute vector from
    ``encoder``, in which case the embedding reflects the *joint* of structure
    and attributes rather than structure alone.  Weights use PyTorch's default
    initialisation under a fixed seed.
    """

    def __init__(self, layers: int = 3, hidden: int = 35, seed: int = 0,
                 encoder=None):
        self.layers, self.hidden, self.seed = layers, hidden, seed
        self.encoder = encoder                     # optional attribute encoder
        self._net = None

    @property
    def name(self) -> str:
        return "gin-attributed" if self.encoder else "gin"

    def _build(self, in_dim: int):
        import torch
        import torch.nn as nn
        with torch.random.fork_rng():
            torch.manual_seed(self.seed)
            proj = nn.Linear(in_dim, self.hidden)
            gins = nn.ModuleList([
                nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                              nn.Linear(self.hidden, self.hidden), nn.ReLU())
                for _ in range(self.layers)])
        for p in list(proj.parameters()) + list(gins.parameters()):
            p.requires_grad_(False)
        self._net = (proj, gins)

    def __call__(self, feeders: Sequence[Feeder]) -> np.ndarray:
        import torch
        rows = []
        for f in feeders:
            u = torch.as_tensor(_undirected(f.adjacency), dtype=torch.float32)
            x = (torch.as_tensor(self.encoder(f), dtype=torch.float32) if self.encoder
                 else u.sum(1, keepdim=True))                          # degree
            if self._net is None:
                self._build(x.shape[1])
            proj, gins = self._net
            with torch.no_grad():
                h = proj(x)
                pooled = []
                for gin in gins:
                    h = gin(h + u @ h)                                 # sum aggregation
                    pooled.append(h.sum(0))                            # sum readout
            rows.append(torch.cat(pooled).numpy())                     # jumping knowledge
        return np.stack(rows).astype(np.float64)


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


# ── summary score ────────────────────────────────────────────────────────────
def realism_score(precision: float, recall: float,
                  ref_precision: float, ref_recall: float) -> float:
    """Realism on a 0-10 scale: F1 of precision and recall, relative to real vs real.

    The harmonic mean drops to zero when either term does, so high precision
    cannot hide a collapse in recall.  Dividing by the real-vs-real F1 measured
    with the same embedding and sample size anchors the scale: 10 is what an
    independent real sample reaches, 0 is an untrained model.  FD is left out
    on purpose, because its scale depends on the embedding initialisation.
    """
    def f1(p, r):
        return 0.0 if p + r <= 0 else 2.0 * p * r / (p + r)
    ref = f1(ref_precision, ref_recall)
    return 0.0 if ref <= 0 else 10.0 * f1(precision, recall) / ref


# ── tree check ───────────────────────────────────────────────────────────────
def is_radial_tree(adjacency: np.ndarray) -> bool:
    """Connected, acyclic, single-rooted: one node without a parent, every other
    node with exactly one, and every node reachable from the root."""
    a = np.asarray(adjacency) != 0
    m = a.shape[0]
    in_degree = a.sum(0)
    roots = np.flatnonzero(in_degree == 0)
    if len(roots) != 1 or (np.delete(in_degree, roots) != 1).any():
        return False
    seen, frontier = {int(roots[0])}, [int(roots[0])]
    while frontier:
        children = np.flatnonzero(a[frontier].any(0))
        frontier = [int(c) for c in children if int(c) not in seen]
        seen.update(frontier)
    return len(seen) == m


# ── node attributes as GIN input ─────────────────────────────────────────────
class AttributeEncoder:
    """Per-node numeric vector for the attributed embedding.

    ``length`` is log10-scaled and standardised, ``ampacity`` one-hot over the
    observed ratings (nearest match), ``phase`` one-hot over the seven codes.
    Fit it on **real** feeders and use the same instance for every generated
    set, so all models are embedded identically.
    """

    def __init__(self, ampacities, log_mean: float, log_std: float,
                 n_phase: int = 7, eps: float = 1e-3):
        self.ampacities = np.asarray(ampacities, dtype=float)
        self.log_mean, self.log_std = float(log_mean), float(log_std)
        self.n_phase, self.eps = n_phase, eps

    @property
    def dim(self) -> int:
        return 1 + len(self.ampacities) + self.n_phase

    @classmethod
    def fit(cls, feeders: Sequence[Feeder], eps: float = 1e-3) -> "AttributeEncoder":
        length = np.concatenate([np.asarray(f.attributes["length"], float) for f in feeders])
        amps = np.concatenate([np.asarray(f.attributes["ampacity"], float) for f in feeders])
        log_length = np.log10(np.clip(length, 0.0, None) + eps)
        return cls(np.unique(np.round(amps, 3)), log_length.mean(),
                   log_length.std() + 1e-8, eps=eps)

    def __call__(self, feeder: Feeder) -> np.ndarray:
        length = np.asarray(feeder.attributes["length"], float)
        amps = np.asarray(feeder.attributes["ampacity"], float)
        phase = np.clip(np.asarray(feeder.attributes["phase"]).astype(int), 0, self.n_phase - 1)
        log_length = (np.log10(np.clip(length, 0.0, None) + self.eps)
                      - self.log_mean) / self.log_std
        amp_idx = np.abs(amps[:, None] - self.ampacities[None, :]).argmin(1)
        return np.concatenate([log_length[:, None],
                               np.eye(len(self.ampacities))[amp_idx],
                               np.eye(self.n_phase)[phase]], axis=1)


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
    """Metrics over the joint of structure and attributes.

    ``encoder`` maps a feeder to per-node vectors, normally an
    :class:`AttributeEncoder` fitted on the real feeders.
    """
    gin = RandomGINEmbedding(seed=seed, encoder=encoder)
    return [
        DescriptorMetric(gin, FrechetDistance()),
        DescriptorMetric(gin, PrecisionRecallDensityCoverage()),
    ]
