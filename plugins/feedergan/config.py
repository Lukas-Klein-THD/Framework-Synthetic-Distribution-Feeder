"""
Configuration for the reference plugin.

Everything the plugin needs is declared here so that a reader can see the whole
setup in one place, and so that no other module hard-codes a path or a
hyperparameter.  Values may be overridden by environment variable or by passing
keyword arguments to the component constructors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# ── data ─────────────────────────────────────────────────────────────────────
#: Directory of preprocessed ``.pt`` graphs.  The plugin reads data from here but
#: contains no code dependency on the project that produced it.
DATA_DIR = os.environ.get(
    "GRIDGEN_DATA_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "..", "paper-implementation", "tensors"))

#: Where trained weights are written.
CHECKPOINT_DIR = os.environ.get(
    "GRIDGEN_CHECKPOINT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "checkpoints"))

#: Fixed graph size.  Graphs with fewer nodes are dropped; larger ones are
#: truncated to their first M BFS nodes, which is a valid rooted subtree because
#: BFS ordering guarantees that a parent precedes its children.
NUM_NODES = 128

#: Device-type taxonomy, in class order.
DEVICE_TYPES = ("line", "switch", "fuse", "recloser")

#: Phase configurations, in class order; the index is the stored phase code.
PHASE_LABELS = ("a", "b", "c", "ab", "bc", "ac", "abc")


@dataclass(frozen=True)
class TopologyConfig:
    """Stage-1 adversarial topology model."""

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
    seed: int = 0


@dataclass(frozen=True)
class AttributeConfig:
    """Stage-2 conditional mixture density model."""

    hidden: int = 128
    gin_layers: int = 4
    n_components: int = 8
    sigma_min: float = 1e-2
    sigma_max: float = 2.0
    cond_dim: int = 16
    parent_dim: int = 16
    profile_dim: int = 6
    lr: float = 1e-4
    batch_size: int = 16
    iterations: int = 6000
    grad_clip: float = 5.0
    lambda_balance: float = 0.05
    val_fraction: float = 0.1
    seed: int = 0


@dataclass(frozen=True)
class PowerFlowConfig:
    """OpenDSS snapshot solve."""

    base_kv: float = 12.47
    kw_per_node: float = 100.0
    power_factor: float = 0.95
    length_units: str = "kft"
    linecodes: str = field(default_factory=lambda: os.environ.get(
        "GRIDGEN_LINECODES", ""))
    ansi_a: tuple[float, float] = (0.95, 1.05)
    ansi_b: tuple[float, float] = (0.9167, 1.0583)
    unbalance_limit_pct: float = 3.0
