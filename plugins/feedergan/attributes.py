"""
Stage 2 — conditional attribute generation by maximum likelihood.

A shared encoder reads the given topology; one head per attribute then models its
distribution.  Head types are chosen to match each attribute's statistics:

    device_type   softmax          generated first, conditions the others
    length        Gaussian mixture multimodal (device placeholders vs conductors)
    ampacity      gated Markov     ~97% of consecutive conductors share a rating
    phase         gated Markov     phases persist along runs, changing at branches

The gated head is the piece that matters most.  It predicts a Bernoulli *change
gate* alongside a class distribution: unless the gate fires, the node copies its
parent's value exactly.  An independent softmax leaks a change on nearly every
edge and destroys the runs that real feeders exhibit.

A six-dimensional :class:`FeederProfile` summarising the feeder as a whole is
broadcast to every node, so attributes are coherent at feeder level rather than
each node drifting to the population mean.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gridgen.core.feeder import FeederSchema
from gridgen.core.registry import register
from gridgen.generation.base import AttributeGenerator

from plugins.feedergan import config
from plugins.feedergan.data import ATTRIBUTES, schema
from plugins.feedergan.layers import DirectedGINEncoder, gather_parent, parent_index
from plugins.feedergan.profile import FeederProfile

LOG2PI = math.log(2.0 * math.pi)


# ── heads ────────────────────────────────────────────────────────────────────
class MixtureHead(nn.Module):
    """Gaussian mixture over a continuous attribute, modelled in log space."""

    def __init__(self, in_dim: int, cfg: config.AttributeConfig):
        super().__init__()
        self.k = cfg.n_components
        self.sigma_min, self.sigma_max = cfg.sigma_min, cfg.sigma_max
        self.net = nn.Sequential(nn.Linear(in_dim, cfg.hidden), nn.GELU(),
                                 nn.Linear(cfg.hidden, self.k * 3))
        self.register_buffer("mu", torch.tensor(0.0))
        self.register_buffer("sigma", torch.tensor(1.0))
        self.register_buffer("lo", torch.tensor(-10.0))
        self.register_buffer("hi", torch.tensor(10.0))
        self._spread_init()

    def _spread_init(self) -> None:
        last = self.net[-1]
        nn.init.normal_(last.weight, 0.0, 0.01)
        with torch.no_grad():
            bias = last.bias.view(self.k, 3)
            bias[:, 0] = 0.0
            bias[:, 1] = torch.linspace(-2.0, 2.0, self.k)
            frac = min(max((0.3 - self.sigma_min) / (self.sigma_max - self.sigma_min),
                           1e-4), 1 - 1e-4)
            bias[:, 2] = math.log(frac / (1 - frac))

    @torch.no_grad()
    def fit(self, values: torch.Tensor) -> None:
        t = torch.log(values.clamp_min(0.0) + 1e-3)
        self.mu.copy_(t.mean())
        self.sigma.copy_(t.std().clamp_min(1e-6))
        self.lo.copy_(torch.quantile(t, 0.001))
        self.hi.copy_(torch.quantile(t, 0.999))

    def _params(self, x):
        p = self.net(x).view(*x.shape[:-1], self.k, 3)
        log_pi = F.log_softmax(p[..., 0], dim=-1)
        sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * torch.sigmoid(p[..., 2])
        return log_pi, p[..., 1], sigma

    def loss(self, x, raw, lambda_balance=0.0):
        log_pi, mu, sigma = self._params(x)
        y = ((torch.log(raw.clamp_min(0.0) + 1e-3) - self.mu) / self.sigma).unsqueeze(-1)
        comp = -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * LOG2PI
        nll = -torch.logsumexp(log_pi + comp, dim=-1).mean()
        balance = torch.zeros((), device=x.device)
        if lambda_balance > 0:
            mean_pi = log_pi.exp().mean(dim=(0, 1)).clamp_min(1e-8)
            balance = math.log(self.k) + (mean_pi * mean_pi.log()).sum()
        return nll + lambda_balance * balance

    @torch.no_grad()
    def sample(self, x):
        log_pi, mu, sigma = self._params(x)
        shape = mu.shape[:-1]
        flat_pi = log_pi.exp().reshape(-1, self.k)
        idx = torch.multinomial(flat_pi, 1).squeeze(-1)
        rows = torch.arange(flat_pi.size(0), device=x.device)
        m = mu.reshape(-1, self.k)[rows, idx]
        s = sigma.reshape(-1, self.k)[rows, idx]
        t = (m + s * torch.randn_like(m)).view(shape) * self.sigma + self.mu
        return (torch.exp(t.clamp(float(self.lo), float(self.hi))) - 1e-3).clamp_min(0.0)


class CategoricalHead(nn.Module):
    """Softmax over the observed values of a discrete attribute."""

    def __init__(self, in_dim: int, cfg: config.AttributeConfig, values=None):
        super().__init__()
        self.in_dim, self.hidden = in_dim, cfg.hidden
        self.net: nn.Module | None = None
        self.register_buffer("values", torch.zeros(0))
        if values is not None:
            self._build(len(values))
            self.values.copy_(torch.tensor(values, dtype=torch.float32))

    def _build(self, n_classes: int) -> None:
        self.net = nn.Sequential(nn.Linear(self.in_dim, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, n_classes))
        if self.values.numel() != n_classes:
            self.register_buffer("values", torch.zeros(n_classes))

    @property
    def n_classes(self) -> int:
        return int(self.values.numel())

    @torch.no_grad()
    def fit(self, raw: torch.Tensor) -> None:
        if self.n_classes:
            return
        observed = torch.unique(raw).sort().values
        self._build(observed.numel())
        self.values.copy_(observed)

    def to_index(self, raw):
        return (raw.unsqueeze(-1) - self.values).abs().argmin(dim=-1)

    def loss(self, x, raw, lambda_balance=0.0):
        logits = self.net(x)
        return F.cross_entropy(logits.reshape(-1, self.n_classes),
                               self.to_index(raw).reshape(-1))

    @torch.no_grad()
    def sample(self, x):
        probs = self.net(x).softmax(-1)
        idx = torch.multinomial(probs.reshape(-1, self.n_classes), 1).squeeze(-1)
        return self.values[idx].view(*x.shape[:-1])


class GatedMarkovHead(CategoricalHead):
    """Discrete attribute with an explicit parent-copy mechanism.

    Predicts ``P(value changes from the parent)`` and a class distribution.  When
    the gate does not fire the parent's value is copied exactly, which is what
    reproduces the long runs of constant rating and phase seen in real feeders.
    """

    def _build(self, n_classes: int) -> None:
        self.net = nn.Sequential(nn.Linear(self.in_dim, self.hidden), nn.GELU(),
                                 nn.Linear(self.hidden, n_classes + 1))
        if self.values.numel() != n_classes:
            self.register_buffer("values", torch.zeros(n_classes))

    def forward_parts(self, x):
        out = self.net(x)
        return out[..., 0], out[..., 1:]              # gate logit, class logits

    def loss(self, x, raw, parent_class, lambda_balance=0.0):
        gate_logit, class_logits = self.forward_parts(x)
        cls = self.to_index(raw)
        changed = (cls != parent_class).float()
        gate = F.binary_cross_entropy_with_logits(gate_logit, changed)
        mask = changed.reshape(-1) > 0.5
        if not mask.any():
            return gate
        cross = F.cross_entropy(class_logits.reshape(-1, self.n_classes)[mask],
                                cls.reshape(-1)[mask])
        return gate + cross


# ── the model ────────────────────────────────────────────────────────────────
class AttributeModel(nn.Module):
    """Shared encoder plus one head per attribute."""

    ORDER = ("device_type", "length", "ampacity", "phase")
    MARKOV = ("ampacity", "phase")

    def __init__(self, cfg: config.AttributeConfig | None = None):
        super().__init__()
        self.cfg = cfg or config.AttributeConfig()
        self.encoder = DirectedGINEncoder(self.cfg.hidden, self.cfg.gin_layers)
        enc = self.encoder.out_dim + self.cfg.profile_dim
        self.type_embed = nn.Embedding(len(config.DEVICE_TYPES), self.cfg.cond_dim)
        self.parent_embed = nn.ModuleDict()

        self.heads = nn.ModuleDict({
            "device_type": CategoricalHead(
                enc, self.cfg, values=list(range(len(config.DEVICE_TYPES)))),
            "length": MixtureHead(enc + self.cfg.cond_dim, self.cfg),
            "ampacity": GatedMarkovHead(enc + 2 * self.cfg.cond_dim + self.cfg.parent_dim,
                                        self.cfg),
            "phase": GatedMarkovHead(enc + 2 * self.cfg.cond_dim + self.cfg.parent_dim,
                                     self.cfg),
        })

    # ── setup ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def fit(self, attributes: dict[str, torch.Tensor]) -> None:
        """Learn class sets and normalisation from real values."""
        for name in self.ORDER:
            self.heads[name].fit(attributes[name].reshape(-1))
        for name in self.MARKOV:
            n = self.heads[name].n_classes
            self.parent_embed[name] = nn.Embedding(n + 1, self.cfg.parent_dim)

    def _context(self, adjacency, profile):
        h = self.encoder(adjacency)
        if self.cfg.profile_dim:
            h = torch.cat([h, profile.unsqueeze(1).expand(-1, h.size(1), -1)], dim=-1)
        return h

    # ── training ─────────────────────────────────────────────────────────────
    def loss(self, adjacency, attributes, profile):
        h = self._context(adjacency, profile)
        parents = parent_index(adjacency)
        total = torch.zeros((), device=adjacency.device)
        parts: dict[str, float] = {}

        total_dt = self.heads["device_type"].loss(h, attributes["device_type"])
        total = total + total_dt
        parts["device_type"] = float(total_dt)

        type_idx = self.heads["device_type"].to_index(attributes["device_type"])
        type_vec = self.type_embed(type_idx)                       # teacher forcing
        parent_type = gather_parent(type_vec, parents)

        length_loss = self.heads["length"].loss(
            torch.cat([h, type_vec], -1), attributes["length"], self.cfg.lambda_balance)
        total = total + length_loss
        parts["length"] = float(length_loss)

        for name in self.MARKOV:
            head = self.heads[name]
            cls = head.to_index(attributes[name])
            parent_cls = torch.gather(cls, 1, parents).clone()
            parent_cls[:, 0] = head.n_classes                      # root sentinel
            x = torch.cat([h, type_vec, parent_type,
                           self.parent_embed[name](parent_cls)], dim=-1)
            loss = head.loss(x, attributes[name], parent_cls)
            total = total + loss
            parts[name] = float(loss)
        return total, parts

    # ── generation ───────────────────────────────────────────────────────────
    @torch.no_grad()
    def sample(self, adjacency, profile, constrain_phase: bool = False):
        h = self._context(adjacency, profile)
        b, m, _ = h.shape
        parents = parent_index(adjacency)
        out: dict[str, torch.Tensor] = {}

        out["device_type"] = self.heads["device_type"].sample(h)
        type_idx = self.heads["device_type"].to_index(out["device_type"])
        type_vec = self.type_embed(type_idx)
        parent_type = gather_parent(type_vec, parents)
        out["length"] = self.heads["length"].sample(torch.cat([h, type_vec], -1))

        classes = {n: torch.zeros(b, m, dtype=torch.long, device=h.device)
                   for n in self.MARKOV}
        for j in range(m):                                   # BFS order: parent first
            for name in self.MARKOV:
                head = self.heads[name]
                if j == 0:
                    parent_cls = torch.full((b,), head.n_classes, dtype=torch.long,
                                            device=h.device)
                else:
                    parent_cls = classes[name].gather(1, parents[:, j:j + 1]).squeeze(1)
                x = torch.cat([h[:, j], type_vec[:, j], parent_type[:, j],
                               self.parent_embed[name](parent_cls)], dim=-1)
                gate_logit, class_logits = head.forward_parts(x)
                if constrain_phase and name == "phase":
                    class_logits = _mask_phase(class_logits, parent_cls, head.n_classes)
                change = torch.bernoulli(torch.sigmoid(gate_logit)).bool() | (j == 0)
                drawn = torch.multinomial(class_logits.softmax(-1), 1).squeeze(1)
                classes[name][:, j] = torch.where(change, drawn, parent_cls)
        for name in self.MARKOV:
            out[name] = self.heads[name].values[classes[name]]
        return out


_PHASE_BITS = torch.tensor([1, 2, 4, 3, 6, 5, 7], dtype=torch.long)


def _mask_phase(logits, parent_class, n_classes):
    """Restrict phases to subsets of the parent's — validity by construction."""
    bits = _PHASE_BITS.to(logits.device)[:n_classes]
    allowed = (bits.view(1, -1) & ~bits.view(-1, 1)) == 0
    allowed = torch.cat([allowed, torch.ones(1, n_classes, dtype=torch.bool,
                                             device=logits.device)], dim=0)
    return logits.masked_fill(~allowed[parent_class], float("-inf"))


@register("attribute_generator", "mdn")
class MDNAttributes(AttributeGenerator):
    """Framework adapter around the trained attribute model."""

    def __init__(self, model: AttributeModel | None = None,
                 profile: FeederProfile | None = None,
                 checkpoint: str | None = None, device: str = "cpu",
                 constrain_phase: bool = False):
        self.device = device
        self.constrain_phase = constrain_phase
        self.model = (model or AttributeModel()).to(device)
        self.profile = profile
        if checkpoint:
            self.load(checkpoint)
        self.model.eval()

    def load(self, path: str) -> "MDNAttributes":
        """Rebuild from a checkpoint.

        Head sizes are data-dependent (the ampacity class set is learned), so the
        heads are re-created at the stored sizes before the weights are loaded.
        """
        blob = torch.load(path, map_location=self.device, weights_only=True)
        cfg = config.AttributeConfig(**blob["config"]) if "config" in blob \
            else config.AttributeConfig()
        self.model = AttributeModel(cfg)
        for name, n_classes in blob["n_classes"].items():
            self.model.heads[name]._build(n_classes)
            self.model.parent_embed[name] = nn.Embedding(n_classes + 1, cfg.parent_dim)
        self.model.load_state_dict(blob["state_dict"])
        self.model.to(self.device).eval()
        self.profile = FeederProfile.from_state(blob["profile"])
        return self

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(self.profile.available_categories) if self.profile else ()

    @property
    def schema(self) -> FeederSchema:
        return schema()

    @torch.no_grad()
    def generate(self, adjacencies, *, category=None, rng=None):
        self._check_category(category)
        A = torch.as_tensor(np.stack(adjacencies), dtype=torch.float32, device=self.device)
        profile = self.profile.sample(A.size(0), device=self.device, category=category)
        out = self.model.sample(A, profile, constrain_phase=self.constrain_phase)
        return [{k: v[i].cpu().numpy() for k, v in out.items()} for i in range(A.size(0))]
