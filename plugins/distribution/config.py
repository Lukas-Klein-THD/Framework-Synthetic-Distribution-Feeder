"""
Configuration shared by every model plugin: data, graph size, evaluation.

Model hyperparameters live in each model's own plugin; this module holds only
what the models and the evaluation must agree on.  Values may be overridden by
environment variable or by passing keyword arguments to the components.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── data ─────────────────────────────────────────────────────────────────────
#: Directory of preprocessed ``.pt`` graphs (SMART-DS / GSO / SAF feeders).  The
#: plugin reads data from here but has no code dependency on the project that
#: produced it.  Each node is a line segment; switching and protection devices
#: are merged into the segments they connect, and lengths are in km.
DATA_DIR = os.environ.get(
    "GRIDGEN_DATA_DIR",
    os.path.join(_REPO, "..", "paper-implementation", "tensors_v2"))

#: Root for trained weights; each model plugin writes to its own subdirectory.
CHECKPOINT_ROOT = os.environ.get("GRIDGEN_CHECKPOINT_DIR", os.path.join(_REPO, "checkpoints"))

#: Fixed graph size.  Graphs with fewer nodes are dropped; larger ones are
#: truncated to their first M BFS nodes, which is a valid rooted subtree because
#: BFS ordering guarantees that a parent precedes its children.
NUM_NODES = 128

#: Phase configurations, in class order; the index is the stored phase code.
PHASE_LABELS = ("a", "b", "c", "ab", "bc", "ac", "abc")


@dataclass(frozen=True)
class PowerFlowConfig:
    """OpenDSS snapshot solve.

    The operating point comes from the SMART-DS models themselves
    (custom-approach ``analyze_load_level.py``):
    - The median feeder carries 20 kW per segment.
    - The truncated M-node tree is the feeder head, so its trunk carries the
      whole feeder's load.
    - A uniform 50 kW per node reproduces that tree's realistic voltage profile
      and line loading.
    - The SMART-DS sources are regulated to 1.03 pu.
    """

    base_kv: float = 12.47
    source_pu: float = 1.03
    kw_per_node: float = 50.0
    power_factor: float = 0.95
    length_units: str = "km"
    linecodes: str = field(default_factory=lambda: os.environ.get(
        "GRIDGEN_LINECODES", ""))
    ansi_a: tuple[float, float] = (0.95, 1.05)
    ansi_b: tuple[float, float] = (0.9167, 1.0583)
    unbalance_limit_pct: float = 3.0
