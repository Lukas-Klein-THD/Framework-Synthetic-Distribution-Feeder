"""
Electrical constraints on a feeder.

Two rules, deliberately of different kinds, because the distinction matters:

* **Phase subset** is a hard physical rule — a segment cannot carry a phase its
  parent does not supply.  Real feeders satisfy it exactly, so the target is 1.0.
* **Ampacity tapering** is a soft engineering tendency — conductor ratings
  usually decrease downstream, but reconductoring and load pockets legitimately
  break it.  Real feeders satisfy it on about 96 % of edges but on only about
  8 % of feeders, so forcing it to 1.0 would overshoot reality.

Always call :meth:`Constraint.calibrate` on real data before judging generated
feeders: that is what tells you which kind of rule you are looking at, and what
target is appropriate.
"""

from __future__ import annotations

import numpy as np

from gridgen.constraints.base import Constraint
from gridgen.core.feeder import Feeder
from gridgen.core.registry import register

#: Bit masks over {A, B, C} for phase codes a, b, c, ab, bc, ac, abc.
PHASE_BITS = np.array([1, 2, 4, 3, 6, 5, 7], dtype=int)
_BITS_TO_CODE = {int(b): i for i, b in enumerate(PHASE_BITS)}


def _parents(feeder: Feeder) -> np.ndarray:
    return np.asarray(feeder.adjacency).argmax(axis=0)


@register("constraint", "phase-subset")
class PhaseSubset(Constraint):
    """A segment's phase set must be a subset of its parent's."""

    hard = True

    @property
    def name(self) -> str:
        return "phase-subset"

    def check(self, feeder: Feeder) -> np.ndarray:
        parents = _parents(feeder)
        code = np.clip(np.asarray(feeder.attributes["phase"]).astype(int), 0, 6)
        bits = PHASE_BITS[code]
        return (bits[1:] & ~bits[parents[1:]]) == 0        # root has no parent edge

    def repair(self, feeder: Feeder) -> Feeder:
        """Clamp each child to the intersection with its parent, root outward.

        Repair edits already-generated values and therefore moves the
        distribution; it reaches validity, but not for free.
        """
        parents = _parents(feeder)
        code = np.clip(np.asarray(feeder.attributes["phase"]).astype(int), 0, 6).copy()
        for j in range(1, len(code)):
            parent_bits = PHASE_BITS[code[int(parents[j])]]
            child_bits = PHASE_BITS[code[j]]
            if child_bits & ~parent_bits:
                shared = child_bits & parent_bits
                code[j] = _BITS_TO_CODE[int(shared if shared else parent_bits)]
        return feeder.with_attributes(phase=code.astype(np.float32))


@register("constraint", "ampacity-tapering")
class AmpacityTapering(Constraint):
    """Conductor rating should not increase downstream, over every edge.

    Every node is a line segment, so every parent -> child edge is a conductor
    edge.  (When switches and fuses were nodes, their device ratings had to be
    excluded; merging them into the segments removed that special case.)
    """

    hard = False

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    @property
    def name(self) -> str:
        return "ampacity-tapering"

    def check(self, feeder: Feeder) -> np.ndarray:
        parents = _parents(feeder)
        ampacity = np.asarray(feeder.attributes["ampacity"], dtype=float)
        child = np.arange(1, len(ampacity))
        return ampacity[child] <= ampacity[parents[child]] + self.tolerance
