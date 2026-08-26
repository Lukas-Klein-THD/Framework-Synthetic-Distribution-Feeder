"""
End-to-end run: generate, optionally enforce constraints, evaluate.

The pipeline holds no domain logic.  It exists so that an experiment is described
by *which components* are plugged in, and so that swapping one — a different
generator, an added metric, a new constraint — changes nothing else.

    pipeline = Pipeline(
        source=my_data,
        generator=my_generator,
        constraints=[PhaseSubset()],
        metrics=[degree_mmd, embedding_fid],
    )
    report = pipeline.run(n=512, category="rural")
    print(report)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from gridgen.constraints.base import Constraint, ValidationResult
from gridgen.core.feeder import Feeder
from gridgen.data.base import DataSource
from gridgen.evaluation.base import Metric
from gridgen.evaluation.scorecard import Evaluator, Scorecard
from gridgen.generation.base import FeederGenerator


@dataclass
class Report:
    """Everything one run produced."""

    scorecard: Scorecard | None = None
    validation: list[ValidationResult] = field(default_factory=list)
    calibration: list[ValidationResult] = field(default_factory=list)
    generated: list[Feeder] = field(default_factory=list)
    real: list[Feeder] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        blocks: list[str] = []
        if self.problems:
            blocks.append("schema problems:\n  " + "\n  ".join(self.problems[:10]))
        if self.calibration or self.validation:
            lines = ["\n=== constraints ===",
                     f"{'constraint':<28}{'real elem':>11}{'real feeder':>13}"
                     f"{'gen elem':>11}{'gen feeder':>12}"]
            lines.append("-" * 75)
            cal = {c.name: c for c in self.calibration}
            for v in self.validation:
                c = cal.get(v.name)
                lines.append(
                    f"{v.name:<28}"
                    + (f"{c.element_rate:>11.4f}{c.feeder_rate:>13.4f}" if c
                       else f"{'-':>11}{'-':>13}")
                    + f"{v.element_rate:>11.4f}{v.feeder_rate:>12.4f}")
            blocks.append("\n".join(lines))
        if self.scorecard is not None:
            blocks.append(str(self.scorecard))
        return "\n".join(blocks) if blocks else "(empty report)"


class Pipeline:
    """Wires a data source, a generator, constraints and metrics together."""

    def __init__(self,
                 source: DataSource,
                 generator: FeederGenerator,
                 constraints: Sequence[Constraint] = (),
                 metrics: Sequence[Metric] = ()):
        self.source = source
        self.generator = generator
        self.constraints = tuple(constraints)
        self.metrics = tuple(metrics)

    def run(self, n: int = 128, *, category: str | None = None,
            repair: bool = False, with_reference: bool = True,
            check_schema: bool = True, seed: int | None = 0) -> Report:
        """Generate ``n`` feeders and score them against ``n`` real ones.

        Parameters
        ----------
        category:
            Restricts both the real sample and the generated feeders, so a
            category-conditional generator is compared against its own category.
        repair:
            Apply each repairable constraint to the generated feeders before
            scoring.  Reported separately from the unrepaired validation, since
            repair reaches validity at some distributional cost.
        with_reference:
            Also score real against real, giving every metric its floor.
        """
        source = self.source.with_category(category) if category else self.source
        real = source.sample(n, seed=seed)
        generated = self.generator.sample(n, category=category)

        report = Report(real=real, generated=generated)

        if check_schema:
            schema = self.generator.schema
            for i, feeder in enumerate(generated[: min(len(generated), 32)]):
                for problem in feeder.validate(schema):
                    report.problems.append(f"generated[{i}]: {problem}")

        for constraint in self.constraints:
            report.calibration.append(constraint.calibrate(real))
        if repair:
            generated = [self._repair(f) for f in generated]
            report.generated = generated
        for constraint in self.constraints:
            report.validation.append(constraint.validate(generated))

        if self.metrics:
            title = f"{type(self.generator).__name__}" + (f" [{category}]" if category else "")
            report.scorecard = Evaluator(self.metrics).run(
                real, generated, with_reference=with_reference, title=title, seed=seed)
        return report

    def _repair(self, feeder: Feeder) -> Feeder:
        for constraint in self.constraints:
            if constraint.repairable:
                feeder = constraint.repair(feeder)
        return feeder
