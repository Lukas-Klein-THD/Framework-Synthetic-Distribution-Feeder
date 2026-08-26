"""
Generation interfaces.

The central requirement is that a model generating *everything at once* and a
pipeline generating topology and attributes *in stages* must be interchangeable.
That is achieved with one outward-facing interface and three inward-facing ones:

    FeederGenerator      produces complete feeders           <- what callers use
      TopologyGenerator      graph only
      AttributeGenerator     attributes given a graph
      DerivedAttributes      quantities computed, not learned

A monolithic model implements :class:`FeederGenerator` directly.  A staged
pipeline implements the three narrow interfaces and is composed into a
:class:`FeederGenerator` by :class:`~gridgen.generation.staged.StagedGenerator`.
Callers cannot tell the difference, which is the point.

Category-conditional generation is expressed by the optional ``category``
argument.  A generator that ignores categories simply does not use it; one that
supports them advertises which via :attr:`Generator.categories`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

import numpy as np

from gridgen.core.feeder import Feeder, FeederSchema, GraphSpec


class Generator(ABC):
    """Common behaviour: every generator declares what it produces."""

    #: Categories this generator can be conditioned on; empty means unconditional.
    categories: tuple[str, ...] = ()

    @property
    @abstractmethod
    def schema(self) -> FeederSchema:
        """Graph shape and attributes this generator produces."""

    def supports(self, category: str | None) -> bool:
        return category is None or category in self.categories

    def _check_category(self, category: str | None) -> None:
        if not self.supports(category):
            raise ValueError(
                f"{type(self).__name__} cannot condition on category {category!r}; "
                f"supported: {self.categories or '(none)'}")


class FeederGenerator(Generator):
    """Produces complete feeders — graph and attributes together.

    This is the interface evaluation and export code depends on.  Implement it
    directly for a model that generates everything jointly (a diffusion model, a
    monolithic GAN); use :class:`StagedGenerator` to build one out of stages.
    """

    @abstractmethod
    def sample(self, n: int, *, category: str | None = None,
               rng: np.random.Generator | None = None) -> list[Feeder]:
        """Return ``n`` generated feeders."""


class TopologyGenerator(Generator):
    """Produces graphs only.

    Kept separate so a topology model can be developed, evaluated and swapped
    without involving attribute generation at all.
    """

    @property
    def schema(self) -> FeederSchema:
        return FeederSchema(self.graph_spec, ())

    @property
    @abstractmethod
    def graph_spec(self) -> GraphSpec:
        """The kind of graph produced (radial or meshed, size, rooting)."""

    @abstractmethod
    def sample_topology(self, n: int, *, category: str | None = None,
                        rng: np.random.Generator | None = None) -> list[np.ndarray]:
        """Return ``n`` adjacency matrices."""


class AttributeGenerator(Generator):
    """Produces per-node attributes for graphs it is given.

    Taking topologies as input (rather than generating them) is what allows
    attribute quality to be measured independently of topology quality — by
    running the same generator on real graphs.
    """

    @abstractmethod
    def generate(self, adjacencies: Sequence[np.ndarray], *,
                 category: str | None = None,
                 rng: np.random.Generator | None = None
                 ) -> list[Mapping[str, np.ndarray]]:
        """Return one attribute mapping per input adjacency."""


class DerivedAttributes(ABC):
    """Attributes computed deterministically from a finished feeder.

    Distance-to-source, depth, downstream counts and similar quantities are
    functions of the graph and the generated attributes, so learning them invites
    inconsistency.  Computing them keeps them exact by construction.
    """

    @property
    @abstractmethod
    def produces(self) -> tuple[str, ...]:
        """Names of the attributes this component adds."""

    @abstractmethod
    def compute(self, feeder: Feeder) -> Mapping[str, np.ndarray]:
        """Return the derived attributes for one feeder."""
