"""
Running a set of metrics and reporting the result.

The scorecard caches descriptor outputs, so metrics that share a descriptor (an
expensive graph embedding used by several distances) compute it once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from gridgen.core.feeder import Feeder
from gridgen.evaluation.base import (
    Descriptor, DescriptorMetric, Direction, Metric, ReferenceMetric)


@dataclass
class Scorecard:
    """Scores for one comparison, with directions and optional references."""

    scores: dict[str, float] = field(default_factory=dict)
    directions: dict[str, Direction] = field(default_factory=dict)
    reference: dict[str, float] = field(default_factory=dict)
    title: str = "scorecard"

    def __str__(self) -> str:
        if not self.scores:
            return f"{self.title}: (empty)"
        width = max(len(k) for k in self.scores) + 2
        has_ref = bool(self.reference)
        lines = [f"\n=== {self.title} ===",
                 f"{'metric':<{width}}{'value':>12}"
                 + (f"{'real-vs-real':>14}" if has_ref else "")
                 + "   better"]
        lines.append("-" * (width + 12 + (14 if has_ref else 0) + 9))
        for key, value in self.scores.items():
            row = f"{key:<{width}}{value:>12.4f}"
            if has_ref:
                ref = self.reference.get(key)
                row += f"{ref:>14.4f}" if ref is not None else f"{'-':>14}"
            row += f"   {self.directions.get(key, Direction.LOWER_IS_BETTER).value}"
            lines.append(row)
        return "\n".join(lines)


class Evaluator:
    """Runs a collection of metrics over real/generated feeder sets."""

    def __init__(self, metrics: Sequence[Metric]):
        self.metrics = list(metrics)

    def run(self, real: Sequence[Feeder], generated: Sequence[Feeder], *,
            with_reference: bool = True, title: str = "scorecard",
            seed: int | None = 0) -> Scorecard:
        """Score ``generated`` against ``real``.

        ``with_reference`` additionally scores real against real, giving each
        metric the floor (or ceiling) it should be read against.
        """
        card = Scorecard(title=title)
        cache: dict[tuple[int, str], object] = {}

        for metric in self.metrics:
            values = self._compute(metric, real, generated, cache)
            for key, value in values.items():
                name = key if key.startswith(metric.name) else f"{metric.name}.{key}"
                card.scores[name] = float(value)
                card.directions[name] = metric.direction

            if with_reference:
                try:
                    ref = ReferenceMetric(metric).reference(real, seed=seed)
                except Exception:
                    continue
                for key, value in ref.items():
                    name = key if key.startswith(metric.name) else f"{metric.name}.{key}"
                    card.reference[name] = float(value)
        return card

    @staticmethod
    def _compute(metric: Metric, real: Sequence[Feeder], generated: Sequence[Feeder],
                 cache: dict) -> Mapping[str, float]:
        """Compute a metric, reusing descriptor output across metrics that share one."""
        if not isinstance(metric, DescriptorMetric):
            return metric.compute(real, generated)

        desc: Descriptor = metric.descriptor
        key_r, key_g = (id(desc), "real"), (id(desc), "generated")
        if key_r not in cache:
            cache[key_r] = desc(real)
            cache[key_g] = desc(generated)
        return metric.distance(cache[key_r], cache[key_g])  # type: ignore[arg-type]
