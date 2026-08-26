"""
Stage 1 — adversarial topology generation.

The generator builds a rooted tree autoregressively: node *j* attends over the
already-placed nodes and selects exactly one parent.  Radiality is therefore a
property of the construction, not something learned or repaired — every sample
is a connected, acyclic, single-rooted tree with ``M-1`` edges.

Because no parameter is indexed by node count, the same weights generate graphs
of any size; the fixed ``M`` comes from the data pipeline, not the model.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridgen.core.feeder import GraphSpec
from gridgen.core.registry import register
from gridgen.generation.base import TopologyGenerator

from plugins.feedergan import config
from plugins.feedergan.layers import bfs_position_encoding


class TreeGenerator(nn.Module):
    """Noise -> soft parent-assignment matrix (upper triangular)."""

    def __init__(self, cfg: config.TopologyConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.embed = nn.Sequential(nn.Linear(cfg.noise_dim, h), nn.LayerNorm(h))
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=cfg.heads, dim_feedforward=2 * h,
            batch_first=True, norm_first=True, dropout=0.0)
        self.body = nn.TransformerEncoder(layer, num_layers=cfg.layers,
                                          enable_nested_tensor=False)
        self.query = nn.Linear(h, h, bias=False)
        self.key = nn.Linear(h, h, bias=False)
        self.out_degree_proj = nn.Linear(1, h)
        self.tau = cfg.tau_end

    def set_temperature(self, tau: float) -> None:
        self.tau = float(tau)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, m, _ = z.shape
        h = self.embed(z) + bfs_position_encoding(m, self.cfg.hidden, z.device)
        h = self.body(h)
        q, k = self.query(h), self.key(h)

        adjacency = torch.zeros(b, m, m, device=z.device)
        out_degree = torch.zeros(b, m, device=z.device)
        for j in range(1, m):
            # candidates are the already-placed nodes, which is what makes the
            # result a tree: exactly one parent, always of lower index
            keys = k[:, :j] + self.out_degree_proj(out_degree[:, :j].unsqueeze(-1))
            scores = torch.bmm(q[:, j:j + 1], keys.transpose(1, 2)).squeeze(1)
            scores = scores / math.sqrt(self.cfg.hidden)
            if self.training:
                choice = F.gumbel_softmax(scores, tau=self.tau, dim=1, hard=False)
            else:
                choice = F.softmax(scores / self.tau, dim=1)
            adjacency[:, :j, j] = choice
            out_degree = out_degree + adjacency[:, :, j].detach()
        return adjacency


class TreeCritic(nn.Module):
    """Scores a (soft) adjacency; consumes structure only, no attributes."""

    def __init__(self, cfg: config.TopologyConfig):
        super().__init__()
        h = cfg.critic_hidden
        self.input = nn.Sequential(nn.Linear(3, h), nn.GELU())
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, h))
            for _ in range(cfg.critic_layers))
        readout = h * (cfg.critic_layers + 1) * 2
        self.head = nn.Sequential(
            nn.Linear(readout, h), nn.GELU(),
            nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        b, m, _ = adjacency.shape
        out_deg = adjacency.sum(2, keepdim=True) / m
        in_deg = adjacency.sum(1, keepdim=True).transpose(1, 2)
        depth = bfs_position_encoding(m, 1, adjacency.device).expand(b, m, 1)
        h = self.input(torch.cat([out_deg, in_deg, depth], dim=-1))
        collected = [h]
        for block in self.blocks:
            h = h + block(torch.cat([h, torch.bmm(adjacency, h)], dim=-1))
            collected.append(h)
        stacked = torch.cat(collected, dim=-1)
        pooled = torch.cat([stacked.mean(1), stacked.max(1).values], dim=-1)
        return self.head(pooled).squeeze(-1)


def harden(adjacency: torch.Tensor) -> torch.Tensor:
    """Soft parent assignment -> binary tree (each column keeps its argmax)."""
    b, m, _ = adjacency.shape
    hard = torch.zeros_like(adjacency)
    for j in range(1, m):
        parent = adjacency[:, :j, j].argmax(dim=1)
        hard[torch.arange(b), parent, j] = 1.0
    return hard


@register("topology_generator", "tree-gan")
class TreeGANTopology(TopologyGenerator):
    """Framework adapter around the trained generator."""

    categories: tuple[str, ...] = ()

    def __init__(self, cfg: config.TopologyConfig | None = None,
                 checkpoint: str | None = None, device: str = "cpu"):
        self.cfg = cfg or config.TopologyConfig()
        self.device = device
        self.module = TreeGenerator(self.cfg).to(device)
        if checkpoint:
            self.load(checkpoint)
        self.module.eval()

    def load(self, path: str) -> "TreeGANTopology":
        """Rebuild from a checkpoint, including the configuration it was trained with."""
        blob = torch.load(path, map_location=self.device, weights_only=True)
        if "config" in blob:
            self.cfg = config.TopologyConfig(**blob["config"])
            self.module = TreeGenerator(self.cfg).to(self.device)
        self.module.load_state_dict(blob.get("generator", blob))
        self.module.eval()
        return self

    @property
    def graph_spec(self) -> GraphSpec:
        return GraphSpec(directed=True, radial=True, root=0,
                         num_nodes=self.cfg.num_nodes)

    @torch.no_grad()
    def sample_topology(self, n: int, *, category: str | None = None,
                        rng: np.random.Generator | None = None) -> list[np.ndarray]:
        self._check_category(category)
        m = self.cfg.num_nodes
        z = torch.randn(n, m, self.cfg.noise_dim, device=self.device)
        adjacency = harden(self.module(z))
        return [a.cpu().numpy() for a in adjacency]
