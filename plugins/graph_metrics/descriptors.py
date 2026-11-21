"""
Descriptors: feeders -> one vector per feeder.

They read the graph and, optionally, per-node attribute vectors supplied by an
encoder; they know nothing about what the attributes mean.  That is what keeps
them domain-independent: the domain plugin provides the encoder.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gridgen.core.feeder import Feeder
from gridgen.core.registry import register
from gridgen.evaluation.base import Descriptor

from plugins.graph_metrics.graphs import undirected


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
            deg = undirected(f.adjacency).sum(1).astype(int)
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
            u = undirected(f.adjacency)
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
    initialisation under a fixed seed (needs ``torch``).
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
            u = torch.as_tensor(undirected(f.adjacency), dtype=torch.float32)
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
