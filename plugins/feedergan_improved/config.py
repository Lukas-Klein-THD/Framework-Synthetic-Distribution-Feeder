"""
Hyperparameters of the improved FeederGAN.

Every value reproduces the thesis model M1+ (run ``v2_cond_seed0``): the paper's
generator with straight-through Gumbel-softmax categories and ampacity drawn
given phase, a critic with sum aggregation, degree features and mean + max
readout, trained with WGAN-GP.  Data location and graph size are shared by every
model and live in :mod:`plugins.distribution.config`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from plugins.distribution.config import CHECKPOINT_ROOT

#: Where this model's trained weights are written.
CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "feedergan_improved")


@dataclass(frozen=True)
class GeneratorConfig:
    """FeederGAN generator (paper Fig. 6, Eq. 8-10)."""

    noise_dim: int = 20
    latent_dim: int = 128            # L
    categorical_dim: int = 64        # V
    adjacency_dim: int = 64          # M in A = col-softmax(M M^T)
    hard_categorical: bool = True    # straight-through Gumbel-softmax
    #: {head: head} -- the first head's logits also see the second head's sample.
    #: Head order is (level, phase, ampacity), so {2: 1} draws ampacity given phase.
    condition_on: dict = field(default_factory=lambda: {2: 1})


@dataclass(frozen=True)
class CriticConfig:
    """FeederGAN critic (paper Eq. 11-13) with the M1+ changes."""

    embedding_dims: tuple[int, ...] = (16, 8, 8)   # level, phase, ampacity
    gcn1_dim: int = 128
    gcn2_dim: int = 64
    negative_slope: float = 0.2
    sum_aggregation: bool = True     # (A + I) instead of D^-1 (A + I)
    degree_features: bool = True     # log(1 + out-degree), in-degree per node
    readout: str = "mean_max"        # "max" is the paper's readout


@dataclass(frozen=True)
class TrainConfig:
    """WGAN-GP training, as in the thesis model."""

    iterations: int = 5000
    batch_size: int = 1              # one real graph per critic step
    n_critic: int = 5
    lr: float = 1e-4
    betas: tuple[float, float] = (0.0, 0.9)
    lambda_gp: float = 10.0
    checkpoint_interval: int = 500
    seed: int = 0
