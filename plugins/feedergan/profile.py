"""
Feeder-level conditioning.

Generating each node from local structure alone makes every synthetic feeder
drift toward the population mean: the marginals match, but the variety *between*
feeders collapses.  A :class:`FeederProfile` is a short, deterministic summary of
a whole feeder that is broadcast to every node, restoring that variety.

It is an **observed** conditioner, not a latent variable: during training it is
computed from the real feeder, so the model cannot ignore it the way it would
ignore uninformative noise.  At generation there is no real feeder, so a profile
is drawn from a pool of real ones.  Labelling that pool by category makes
generation category-conditional at no additional training cost.

Note for evaluation: the profile summarises the attributes being generated, so
agreement on the statistics it encodes is partly circular.  Quality claims should
rest on quantities it does not encode — persistence, constraint validity, and the
attributed joint distribution.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np
import torch

#: Components of the profile, in vector order.
PROFILE_FIELDS = ("switch_fraction", "fuse_fraction", "log_length_mean",
                  "log_length_std", "log_ampacity_mean", "phase_entropy")


def compute_raw(attributes: dict[str, np.ndarray], n_phase: int = 7,
                eps: float = 1e-3) -> np.ndarray:
    """Un-standardised profile for one feeder."""
    device_type = np.asarray(attributes["device_type"])
    length = np.clip(np.asarray(attributes["length"]), 0.0, None)
    ampacity = np.clip(np.asarray(attributes["ampacity"]), 0.0, None)
    phase = np.clip(np.asarray(attributes["phase"]).astype(int), 0, n_phase - 1)

    conductor = device_type < 0.5
    log_length = np.log10(length + eps)
    log_ampacity = np.log10(ampacity + eps)
    sel = conductor if conductor.any() else np.ones_like(conductor, dtype=bool)

    counts = np.bincount(phase, minlength=n_phase).astype(float)
    p = counts / max(counts.sum(), 1.0)
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())

    return np.array([
        float((np.abs(device_type - 1) < 0.5).mean()),
        float((np.abs(device_type - 2) < 0.5).mean()),
        float(log_length[sel].mean()),
        float(log_length[sel].std()),
        float(log_ampacity[sel].mean()),
        entropy,
    ], dtype=np.float32)


class FeederProfile:
    """Standardisation plus a pool of real profiles to draw from."""

    def __init__(self, mean, std, pool, categories: Sequence[str | None] | None = None):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-6)
        self.pool = torch.as_tensor(pool, dtype=torch.float32)
        self.categories = list(categories) if categories is not None else None

    @property
    def dim(self) -> int:
        return int(self.pool.shape[1])

    @property
    def available_categories(self) -> list[str]:
        return sorted({c for c in (self.categories or []) if c})

    # ── construction ─────────────────────────────────────────────────────────
    @classmethod
    def fit(cls, feeders) -> "FeederProfile":
        """Fit standardisation and build the pool from real feeders."""
        raw = np.stack([compute_raw(f.attributes) for f in feeders])
        mean, std = raw.mean(0), raw.std(0)
        pool = (raw - mean) / np.maximum(std, 1e-6)
        return cls(mean, std, pool, [f.category for f in feeders])

    def encode(self, feeders) -> torch.Tensor:
        """Standardised profiles for given feeders (teacher forcing)."""
        raw = np.stack([compute_raw(f.attributes) for f in feeders])
        return (torch.from_numpy(raw) - self.mean) / self.std

    # ── generation ───────────────────────────────────────────────────────────
    def sample(self, n: int, device=None, category: str | None = None) -> torch.Tensor:
        """Draw ``n`` profiles, optionally restricted to one category."""
        if category is None:
            idx = torch.randint(0, self.pool.size(0), (n,))
        else:
            if not self.categories:
                raise ValueError("profile pool carries no category labels")
            rows = torch.tensor([i for i, c in enumerate(self.categories)
                                 if c == category], dtype=torch.long)
            if rows.numel() == 0:
                raise ValueError(
                    f"no profiles for category {category!r}; "
                    f"available: {self.available_categories}")
            idx = rows[torch.randint(0, rows.numel(), (n,))]
        out = self.pool[idx]
        return out.to(device) if device is not None else out

    def summary(self) -> str:
        mix = Counter(c for c in (self.categories or []) if c)
        return f"FeederProfile(dim={self.dim}, pool={self.pool.size(0)}, mix={dict(mix)})"

    # ── persistence ──────────────────────────────────────────────────────────
    def state(self) -> dict:
        return {"mean": self.mean, "std": self.std, "pool": self.pool,
                "categories": self.categories}

    @classmethod
    def from_state(cls, state: dict) -> "FeederProfile":
        return cls(state["mean"], state["std"], state["pool"], state.get("categories"))
