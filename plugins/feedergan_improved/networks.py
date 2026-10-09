"""
Generator and critic of the improved FeederGAN.

The generator is the paper's (Fig. 6, Eq. 8-10): per-node noise -> latent ->
numeric attributes, categorical attributes and an adjacency from the symmetric
score ``M M^T`` with a column softmax, so each node holds a distribution over
its parent.  Topology and attributes come out of one forward pass.

Two changes to the generator:

* **Straight-through Gumbel-softmax categories.**  Real categories are one-hot;
  a soft output is trivially recognisable as generated, and the cheapest way to
  hide that is to saturate one class on every node (the phase collapse of the
  original).  Sampling also happens at evaluation, so the class marginal the
  model learned is what it emits.
* **Ampacity given phase.**  The ampacity logits also see the sampled phase, so
  the strong phase -> rating coupling of real feeders does not have to pass
  through the shared latent alone.

The critic is the paper's graph convolutional critic (Eq. 11-13) with three
changes that let it see what the original could not:

* **Sum aggregation**, ``(A + I) X W`` instead of ``D^-1 (A + I) X W``: a mean
  makes a node with many identical children look like a node with two, so the
  original critic cannot count children.
* **Degree features**: log(1 + out-degree) and in-degree per node, computed from
  the adjacency it is given.
* **Mean + max readout**: a max over nodes cannot see population statistics
  such as the share of leaves or the number of roots.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from plugins.feedergan_improved.config import CriticConfig, GeneratorConfig
from plugins.feedergan_improved.encoding import CAT_VOCAB_SIZES, NUM_DIM


class Generator(nn.Module):
    """z [B, M, noise] -> (soft adjacency [B, M, M], numeric [B, M, 3], categorical one-hot)."""

    def __init__(self, cfg: GeneratorConfig | None = None):
        super().__init__()
        self.cfg = cfg or GeneratorConfig()
        c = self.cfg
        self.condition_on = {int(k): int(v) for k, v in c.condition_on.items()}
        if any(src >= dst for dst, src in self.condition_on.items()):
            raise ValueError("a head can only be conditioned on an earlier head")

        self.fc1 = nn.Linear(c.noise_dim, c.latent_dim)                 # latent L
        self.fc2 = nn.Linear(c.latent_dim, NUM_DIM)                     # numeric
        self.fc3 = nn.Linear(c.latent_dim, c.categorical_dim)           # V
        self.fc4 = nn.ModuleList([
            nn.Linear(c.categorical_dim + (CAT_VOCAB_SIZES[self.condition_on[k]]
                                           if k in self.condition_on else 0), size)
            for k, size in enumerate(CAT_VOCAB_SIZES)])
        concat = NUM_DIM + sum(CAT_VOCAB_SIZES)
        self.norm = nn.LayerNorm(concat)
        self.fc5 = nn.Linear(concat, c.adjacency_dim)                   # M

    def _categorical(self, logits):
        if not self.cfg.hard_categorical:
            return F.softmax(logits, dim=-1)
        return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)

    def forward(self, z: torch.Tensor):
        m = z.size(1)
        latent = F.relu(self.fc1(z))
        numeric_logits = self.fc2(latent)
        numeric = torch.tanh(numeric_logits)

        v = F.relu(self.fc3(latent))
        logits, samples = [], []
        for k, head in enumerate(self.fc4):
            x = v if k not in self.condition_on else \
                torch.cat([v, samples[self.condition_on[k]]], dim=-1)
            logits.append(head(x))
            samples.append(self._categorical(logits[-1]))
        categorical = torch.cat(samples, dim=-1)

        # adjacency from the logits (not the samples), as in the paper
        fused = self.norm(torch.cat([numeric_logits] + logits, dim=-1))
        rep = self.fc5(fused)
        scores = torch.bmm(rep, rep.transpose(1, 2))
        eye = torch.eye(m, device=z.device, dtype=torch.bool).unsqueeze(0)
        scores = scores.masked_fill(eye, -1e9)
        adjacency = F.softmax(scores, dim=1)          # column j: distribution over parents
        adjacency = adjacency.masked_fill(eye, 0.0)
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return adjacency, numeric, categorical


class _GraphConv(nn.Module):
    """``A X W`` aggregated over children, with self-loop; sum or mean."""

    def __init__(self, in_dim: int, out_dim: int, sum_aggregation: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.sum_aggregation = sum_aggregation

    def forward(self, adjacency, x):
        support = x @ self.weight
        if self.sum_aggregation:
            return torch.bmm(adjacency, support) + support
        eye = torch.eye(adjacency.size(1), device=adjacency.device,
                        dtype=adjacency.dtype).unsqueeze(0)
        a = adjacency + eye
        return torch.bmm(a / a.sum(-1, keepdim=True).clamp_min(1e-8), support)


class Critic(nn.Module):
    """(categorical one-hot, numeric, adjacency) -> score [B]."""

    def __init__(self, cfg: CriticConfig | None = None):
        super().__init__()
        self.cfg = cfg or CriticConfig()
        c = self.cfg
        if c.readout not in ("max", "mean_max"):
            raise ValueError(f"readout must be 'max' or 'mean_max', got {c.readout}")
        self.embeddings = nn.ModuleList([
            nn.Linear(size, dim, bias=False)
            for size, dim in zip(CAT_VOCAB_SIZES, c.embedding_dims)])
        width = sum(c.embedding_dims) + NUM_DIM
        self.norm = nn.LayerNorm(width)
        r_dim = width + (2 if c.degree_features else 0)
        self.gcn1 = _GraphConv(r_dim, c.gcn1_dim, c.sum_aggregation)
        self.gcn2 = _GraphConv(c.gcn1_dim, c.gcn2_dim, c.sum_aggregation)
        node_dim = r_dim + c.gcn1_dim + c.gcn2_dim
        self.out = nn.Linear(node_dim * (2 if c.readout == "mean_max" else 1), 1)

    def forward(self, categorical, numeric, adjacency):
        parts = torch.split(categorical, list(CAT_VOCAB_SIZES), dim=-1)
        embedded = torch.cat([e(p) for e, p in zip(self.embeddings, parts)], dim=-1)
        r = self.norm(torch.cat([embedded, numeric], dim=-1))
        if self.cfg.degree_features:
            out_degree = adjacency.sum(dim=2, keepdim=True)
            in_degree = adjacency.sum(dim=1).unsqueeze(-1)
            r = torch.cat([r, torch.log1p(out_degree), in_degree], dim=-1)
        h = F.leaky_relu(self.gcn1(adjacency, r), self.cfg.negative_slope)
        i = F.leaky_relu(self.gcn2(adjacency, h), self.cfg.negative_slope)
        nodes = torch.cat([r, h, i], dim=-1)
        if self.cfg.readout == "max":
            return self.out(nodes).max(dim=1).values.squeeze(-1)
        pooled = torch.cat([nodes.mean(dim=1), nodes.max(dim=1).values], dim=-1)
        return self.out(pooled).squeeze(-1)
