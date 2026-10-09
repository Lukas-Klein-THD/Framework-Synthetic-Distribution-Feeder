"""
Shared network building blocks of the staged model.

The topology critic and the attribute model read a feeder's structure through
the same directed GIN, so it lives here.  Nothing in this module is specific to
the framework interfaces; it is plain PyTorch.

Node inputs are purely structural: out-degree, in-degree, depth below the root
and BFS position.  Depth is computed from the (possibly soft) adjacency by
forward substitution, which is exact for a real tree and the expected depth for
a generated soft one, and differentiable either way.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_encoding(seq_len: int, dim: int, device=None) -> torch.Tensor:
    """Sinusoidal encoding of BFS position ``[seq_len, dim]``.

    Analytic rather than learned, so a model trained at one graph size still has
    a meaningful encoding at another.
    """
    pos = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    idx = torch.arange(0, dim, 2, dtype=torch.float32, device=device)
    div = torch.exp(-idx * (math.log(10000.0) / dim))
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].size(1)])
    return pe


def directed_row_normalize(a: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Row-normalise ``A + I`` (binary or soft), so isolated rows stay finite."""
    a = a + torch.eye(a.size(-1), device=a.device, dtype=a.dtype).unsqueeze(0)
    return a / a.sum(dim=-1, keepdim=True).clamp_min(eps)


def compute_soft_depth(adj: torch.Tensor) -> torch.Tensor:
    """Depth per node from an upper-triangular (soft) adjacency ``[B, M, M] -> [B, M]``.

    ``depth[j] = sum_i adj[i, j] * (depth[i] + 1)``; forward substitution works
    because every parent has a lower index than its child.
    """
    b, m, _ = adj.shape
    cols = [adj.new_zeros(b)]
    for j in range(1, m):
        so_far = torch.stack(cols, dim=1)
        cols.append((adj[:, :j, j] * (so_far + 1)).sum(dim=1))
    return torch.stack(cols, dim=1)


def structural_inputs(adj: torch.Tensor) -> torch.Tensor:
    """Out-degree, in-degree, depth and BFS position per node ``[B, M, 4]``."""
    b, m, _ = adj.shape
    out_deg = adj.sum(dim=2, keepdim=True) / m
    in_deg = adj.sum(dim=1, keepdim=True).transpose(1, 2)
    depth = (compute_soft_depth(adj) / m).unsqueeze(-1)
    pos = (torch.arange(m, device=adj.device, dtype=adj.dtype) / m).view(1, m, 1).expand(b, m, 1)
    return torch.cat([out_deg, in_deg, depth, pos], dim=-1)


class DirectedGINLayer(nn.Module):
    """Directed GIN layer (Xu et al. 2019): aggregate from parents and from children separately.

    Symmetrising the adjacency would make the root and a leaf of a chain look
    alike; keeping the two directions apart lets the layer see both depth and
    subtree structure.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 3, out_dim * 2), nn.LayerNorm(out_dim * 2), nn.GELU(),
            nn.Linear(out_dim * 2, out_dim), nn.LayerNorm(out_dim))

    def forward(self, adj_down, adj_up, h):
        return self.mlp(torch.cat([(1.0 + self.eps) * h, torch.bmm(adj_down, h),
                                   torch.bmm(adj_up, h)], dim=-1))


class DirectedGINEncoder(nn.Module):
    """Per-node embeddings ``[B, M, (layers + 1) * hidden]`` with jumping knowledge."""

    def __init__(self, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.hidden, self.layers = hidden, layers
        self.fc_node = nn.Sequential(nn.Linear(4, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gin_layers = nn.ModuleList(DirectedGINLayer(hidden, hidden) for _ in range(layers))

    @property
    def out_dim(self) -> int:
        return self.hidden * (self.layers + 1)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        h = self.fc_node(structural_inputs(adj))
        down = directed_row_normalize(adj.transpose(-1, -2))     # from parents
        up = directed_row_normalize(adj)                         # from children
        collected = [h]
        for gin in self.gin_layers:
            h = gin(down, up, h)
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
