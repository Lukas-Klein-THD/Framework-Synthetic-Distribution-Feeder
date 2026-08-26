"""
Composition of stage-wise components into a single :class:`FeederGenerator`.

This is framework infrastructure, not a model: it contains no learning and no
domain logic.  Its only job is to make a staged pipeline satisfy the same
interface as a monolithic generator, so that callers — and the evaluation stack —
cannot tell them apart.

    generator = StagedGenerator(
        topology=my_topology_model,
        attributes=my_attribute_model,
        derived=[MyDerivedAttributes()],
    )
    feeders = generator.sample(64, category="rural")
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gridgen.core.feeder import Feeder, FeederSchema
from gridgen.generation.base import (
    AttributeGenerator, DerivedAttributes, FeederGenerator, TopologyGenerator)


class StagedGenerator(FeederGenerator):
    """A feeder generator assembled from a topology stage and an attribute stage.

    Parameters
    ----------
    topology:
        Produces the graphs.
    attributes:
        Produces per-node attributes for those graphs.  Optional: leaving it out
        yields topology-only feeders, which is useful while a topology model is
        still being developed.
    derived:
        Deterministic post-stages, applied in order.  Each may use attributes
        added by earlier ones.
    """

    def __init__(self,
                 topology: TopologyGenerator,
                 attributes: AttributeGenerator | None = None,
                 derived: Sequence[DerivedAttributes] = ()):
        self.topology = topology
        self.attributes = attributes
        self.derived = tuple(derived)

    # ── declared behaviour ───────────────────────────────────────────────────
    @property
    def schema(self) -> FeederSchema:
        graph = self.topology.graph_spec
        specs = tuple(self.attributes.schema.attributes) if self.attributes else ()
        return FeederSchema(graph, specs)

    @property
    def categories(self) -> tuple[str, ...]:  # type: ignore[override]
        """Categories supported by *every* stage that is category-aware.

        A staged pipeline can only honour a category if no stage would reject it,
        so the intersection is the honest answer.  Stages that are unconditional
        (empty tuple) impose no restriction.
        """
        sets = [set(s.categories) for s in self._stages() if s.categories]
        if not sets:
            return ()
        return tuple(sorted(set.intersection(*sets)))

    def _stages(self):
        return [s for s in (self.topology, self.attributes) if s is not None]

    # ── generation ───────────────────────────────────────────────────────────
    def sample(self, n: int, *, category: str | None = None,
               rng: np.random.Generator | None = None) -> list[Feeder]:
        self._check_category(category)
        rng = rng or np.random.default_rng()

        adjacencies = self.topology.sample_topology(
            n, category=category if self.topology.categories else None, rng=rng)

        if self.attributes is None:
            feeders = [Feeder(a) for a in adjacencies]
        else:
            attrs = self.attributes.generate(
                adjacencies,
                category=category if self.attributes.categories else None, rng=rng)
            feeders = [Feeder(a, dict(x)) for a, x in zip(adjacencies, attrs)]

        feeders = [self.apply_derived(f) for f in feeders]
        if category is not None:
            feeders = [f.with_metadata(category=category) for f in feeders]
        return feeders

    def apply_derived(self, feeder: Feeder) -> Feeder:
        """Run every derived stage in order, accumulating attributes."""
        for stage in self.derived:
            feeder = feeder.with_attributes(**stage.compute(feeder))
        return feeder

    def __repr__(self) -> str:
        parts = [f"topology={type(self.topology).__name__}"]
        if self.attributes is not None:
            parts.append(f"attributes={type(self.attributes).__name__}")
        if self.derived:
            parts.append("derived=[" + ", ".join(type(d).__name__ for d in self.derived) + "]")
        return f"StagedGenerator({', '.join(parts)})"
