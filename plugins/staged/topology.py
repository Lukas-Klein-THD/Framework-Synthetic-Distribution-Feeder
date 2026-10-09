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

from plugins.staged import config
from plugins.staged.layers import DirectedGINLayer, directed_row_normalize, sinusoidal_encoding, structural_inputs


class TreeGenerator(nn.Module):
    """Noise -> soft parent-assignment matrix (upper triangular).

    A Transformer turns per-node noise plus a BFS position encoding into node
    embeddings; node ``j`` then scores every already-placed node as its parent
    by query-key attention.  Each candidate's key is shifted by a learned
    function of its current out-degree, which gives the model explicit control
    over branching.  That correction starts at zero, so training begins from
    plain attention scoring.
    """

    def __init__(self, cfg: config.TopologyConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden
        self.fc_in = nn.Sequential(nn.Linear(cfg.noise_dim, h), nn.LayerNorm(h))
        layer = nn.TransformerEncoderLayer(
            d_model=h, nhead=cfg.heads, dim_feedforward=4 * h, dropout=0.0,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.layers,
                                                 enable_nested_tensor=False)
        self.W_q = nn.Linear(h, h, bias=False)
        self.W_k = nn.Linear(h, h, bias=False)
        self.out_deg_proj = nn.Linear(1, h)
        nn.init.zeros_(self.out_deg_proj.weight)
        nn.init.zeros_(self.out_deg_proj.bias)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        self.tau = cfg.tau_end

    def set_temperature(self, tau: float) -> None:
        self.tau = float(max(tau, 0.05))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, m, _ = z.shape
        e = self.fc_in(z) + sinusoidal_encoding(m, self.cfg.hidden, z.device).unsqueeze(0)
        h = self.transformer(e)
        q, k = self.W_q(h), self.W_k(h)

        adjacency = torch.zeros(b, m, m, device=z.device)
        out_degree = torch.zeros(b, m, device=z.device)
        for j in range(1, m):
            # candidates are the already-placed nodes, which is what makes the
            # result a tree: exactly one parent, always of lower index
            keys = k[:, :j, :] + self.out_deg_proj(out_degree[:, :j].unsqueeze(-1))
            scores = torch.bmm(q[:, j:j + 1, :], keys.transpose(1, 2)).squeeze(1)
            scores = scores / math.sqrt(self.cfg.hidden)
            if self.training:
                choice = F.gumbel_softmax(scores, tau=self.tau, dim=1, hard=False)
            else:
                choice = F.softmax(scores / self.tau, dim=1)
            adjacency[:, :j, j] = choice
            out_degree = out_degree + adjacency[:, :, j].detach()
        return adjacency


class TreeCritic(nn.Module):
    """Directed-GIN critic over topology alone (structural node inputs only)."""

    def __init__(self, cfg: config.TopologyConfig):
        super().__init__()
        h = cfg.critic_hidden
        self.fc_node = nn.Sequential(nn.Linear(4, h), nn.LayerNorm(h), nn.GELU())
        self.gin_layers = nn.ModuleList(DirectedGINLayer(h, h) for _ in range(cfg.critic_layers))
        readout = 2 * (cfg.critic_layers + 1) * h
        self.fc_out = nn.Sequential(
            nn.Linear(readout, h), nn.LayerNorm(h), nn.GELU(),
            nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        h = self.fc_node(structural_inputs(adj))
        down = directed_row_normalize(adj.transpose(-1, -2))
        up = directed_row_normalize(adj)
        collected = [h]
        for gin in self.gin_layers:
            h = gin(down, up, h)
            collected.append(h)
        stacked = torch.cat(collected, dim=-1)
        pooled = torch.cat([stacked.mean(dim=1), stacked.max(dim=1).values], dim=-1)
        return self.fc_out(pooled).squeeze(-1)


def harden(adjacency: torch.Tensor) -> torch.Tensor:
    """Soft parent assignment -> binary tree: each node keeps its most likely parent.

    This is FeederGAN's Algorithm 2 without the row normalisation, which would
    penalise hubs (a node spreading its probability over many children) against
    nodes with a single child.  For this generator's upper-triangular output it
    reduces to a column argmax and always yields a tree rooted at node 0.
    """
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
