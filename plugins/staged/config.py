"""
Hyperparameters of the staged model.

Data location, graph size and evaluation settings are shared by every model and
live in :mod:`plugins.distribution.config`; this module holds only what is
specific to the staged generator.  Values may be overridden by passing a config
instance to the components.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from plugins.distribution.config import CHECKPOINT_ROOT, NUM_NODES

#: Where this model's trained weights are written.
CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "staged")


@dataclass(frozen=True)
class TopologyConfig:
    """Stage 1 — adversarial topology model."""

    num_nodes: int = NUM_NODES
    noise_dim: int = 32
    hidden: int = 128
    heads: int = 4
    layers: int = 4
    critic_hidden: int = 128
    critic_layers: int = 4
    lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.9)
    n_critic: int = 5
    lambda_gp: float = 10.0
    batch_size: int = 8
    iterations: int = 3000
    tau_start: float = 1.0
    tau_end: float = 0.1
    lambda_branch: float = 0.1
    internal_threshold: float = 0.3     # soft out-degree above which a node counts as internal
    seed: int = 0


@dataclass(frozen=True)
class AttributeConfig:
    """Stage 2 — conditional mixture density model."""

    hidden: int = 128
    gin_layers: int = 4
    n_components: int = 8
    sigma_min: float = 1e-2
    sigma_max: float = 2.0
    parent_dim: int = 16
    profile_dim: int = 4
    lr: float = 1e-4
    batch_size: int = 16
    iterations: int = 6000
    grad_clip: float = 5.0
    lambda_balance: float = 0.05
    val_fraction: float = 0.1
    seed: int = 0
