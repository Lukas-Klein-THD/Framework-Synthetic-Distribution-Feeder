"""
Shared network building blocks.

Both stages read a feeder's structure through the same directed GIN encoder, so
it lives here rather than being duplicated.  Nothing in this module is specific
to a stage or to the framework interfaces — it is plain PyTorch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def bfs_position_encoding(num_nodes: int, dim: int, device=None) -> torch.Tensor:
    """Sinusoidal encoding of BFS index.

    Analytic rather than learned, so a model trained at one graph size still has
    a meaningful encoding at another.
    """
    pos = torch.arange(num_nodes, dtype=torch.float32, device=device).unsqueeze(1)
    i = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    denom = torch.exp(-math.log(10000.0) * i / dim)
    out = torch.zeros(num_nodes, dim, device=device)
    out[:, 0::2] = torch.sin(pos * denom)
    out[:, 1::2] = torch.cos(pos * denom)
    return out


class DirectedGINEncoder(nn.Module):
    """Message passing over a directed rooted tree.

    Each layer aggregates separately from parents (the in-edges) and children
    (the out-edges), because in a feeder the two directions mean different
    things: upstream carries the supply, downstream the load.  A jumping-knowledge
    readout concatenates every layer so both local and multi-hop context reach
    the heads.
    """

    def __init__(self, hidden: int = 128, layers: int = 4, node_in: int = 3):
        super().__init__()
        self.hidden = hidden
        self.layers = layers
        self.input = nn.Sequential(nn.Linear(node_in, hidden), nn.GELU())
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Linear(3 * hidden, hidden), nn.GELU(),
                          nn.Linear(hidden, hidden))
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))

    @property
    def out_dim(self) -> int:
        return self.hidden * (self.layers + 1)

    def forward(self, adjacency: torch.Tensor) -> torch.Tensor:
        """``adjacency`` ``[B, M, M]`` -> node embeddings ``[B, M, out_dim]``."""
        b, m, _ = adjacency.shape
        down = adjacency                                   # parent -> child
        up = adjacency.transpose(1, 2)                     # child -> parent
        out_deg = down.sum(-1, keepdim=True)
        in_deg = up.sum(-1, keepdim=True)
        depth = bfs_position_encoding(m, 1, adjacency.device).expand(b, m, 1)

        h = self.input(torch.cat([out_deg, in_deg, depth], dim=-1))
        collected = [h]
        for block, norm in zip(self.blocks, self.norms):
            agg_down = torch.bmm(down, h)
            agg_up = torch.bmm(up, h)
            h = norm(h + block(torch.cat([h, agg_down, agg_up], dim=-1)))
            collected.append(h)
        return torch.cat(collected, dim=-1)


def parent_index(adjacency: torch.Tensor) -> torch.Tensor:
    """Parent of each node ``[B, M]``; the root maps to itself (index 0)."""
    return adjacency.argmax(dim=1)


def gather_parent(values: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """Value held by each node's parent — ``[B, M, D]`` or ``[B, M]``."""
    if values.dim() == 2:
        return torch.gather(values, 1, parents)
    return torch.gather(values, 1, parents.unsqueeze(-1).expand(-1, -1, values.size(-1)))
