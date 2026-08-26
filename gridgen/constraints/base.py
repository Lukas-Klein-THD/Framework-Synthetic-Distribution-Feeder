"""
Domain-constraint interfaces (the reconstruction stage).

A :class:`Constraint` expresses a rule a feeder should satisfy — phase
propagation, conductor tapering, radiality — and knows two things about it:
whether it holds, and how to make it hold.

Two distinctions are built in because both turned out to matter in practice:

* **Hard versus soft.**  A hard constraint is physically obligatory and real data
  satisfies it exactly; the target is 1.0.  A soft constraint is an engineering
  tendency that real data satisfies only mostly, so forcing it to 1.0 overshoots
  reality.  :attr:`Constraint.hard` records which, and
  :meth:`Constraint.calibrate` measures the real-data rate so the claim can be
  checked rather than assumed.
* **Element-wise versus feeder-wise.**  A rule violated on a small fraction of
  edges can still leave most feeders invalid, since a feeder is valid only if
  every element is.  :class:`ValidationResult` reports both.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from gridgen.core.feeder import Feeder


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of checking one constraint over a set of feeders.

    Attributes
    ----------
    element_rate:
        Fraction of checked elements (edges or nodes) that satisfy the rule.
    feeder_rate:
        Fraction of feeders in which *every* element satisfies it.
    per_feeder:
        Element-wise satisfaction rate for each feeder, for plotting.
    n_elements:
        How many elements were checked in total.
    """

    name: str
    element_rate: float
    feeder_rate: float
    per_feeder: np.ndarray
    n_elements: int

    def summary(self) -> str:
        return (f"{self.name}: {self.element_rate:.4f} element-wise, "
                f"{self.feeder_rate:.4f} feeder-wise ({self.n_elements} elements)")


class Constraint(ABC):
    """A domain rule that can be checked and, optionally, enforced."""

    #: Physically obligatory (target 1.0) rather than a statistical tendency.
    hard: bool = True

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def check(self, feeder: Feeder) -> np.ndarray:
        """Boolean array over the checked elements: ``True`` where satisfied.

        Elements are whatever the rule is about — usually parent/child edges.
        Elements the rule does not apply to should be omitted, not reported as
        satisfied, so the rate is not diluted.
        """

    def repair(self, feeder: Feeder) -> Feeder:
        """Return a feeder satisfying the constraint, editing as little as possible.

        Optional: a constraint that only measures need not implement it.  Repair
        changes already-generated values and can therefore distort the
        distribution, which is why :meth:`validate` reports the rate separately
        from any quality metric.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement repair()")

    @property
    def repairable(self) -> bool:
        return type(self).repair is not Constraint.repair

    # ── aggregation ──────────────────────────────────────────────────────────
    def validate(self, feeders: Sequence[Feeder]) -> ValidationResult:
        """Check the constraint across a set of feeders."""
        per_feeder, ok, total = [], 0, 0
        for f in feeders:
            flags = np.asarray(self.check(f), dtype=bool)
            per_feeder.append(float(flags.mean()) if flags.size else np.nan)
            ok += int(flags.sum())
            total += int(flags.size)
        arr = np.asarray(per_feeder, dtype=float)
        with np.errstate(invalid="ignore"):
            feeder_rate = float(np.nanmean(arr >= 1.0)) if arr.size else float("nan")
        return ValidationResult(
            name=self.name,
            element_rate=float(ok / total) if total else float("nan"),
            feeder_rate=feeder_rate,
            per_feeder=arr,
            n_elements=total,
        )

    def calibrate(self, real: Sequence[Feeder]) -> ValidationResult:
        """Measure the rule on REAL feeders.

        Always do this before reporting a constraint on generated data: it is
        what distinguishes a hard rule (real rate 1.0) from a soft tendency, and
        it establishes the target the generator should actually be judged against.
        """
        return self.validate(real)
