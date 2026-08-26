"""
Evaluation interfaces.

A :class:`Metric` compares a set of real feeders with a set of generated ones and
returns named scores.  Most graph-generation metrics factor into two independent
pieces, so those are separate interfaces:

    Descriptor   feeders -> a fixed-size vector each   (degree histogram, GNN embedding, ...)
    Distance     (real vectors, generated vectors) -> score(s)   (MMD, Frechet, precision/recall)

:class:`DescriptorMetric` combines them, which means a new descriptor works with
every existing distance and vice versa.  A metric that does not factor this way
(anything reading the graph directly) implements :class:`Metric` instead.

Every metric declares a :class:`Direction`, because a scorecard mixing "lower is
better" and "higher is better" columns is otherwise unreadable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from gridgen.core.feeder import Feeder


class Direction(str, Enum):
    """Which way is better for a score."""

    LOWER_IS_BETTER = "lower"
    HIGHER_IS_BETTER = "higher"


class Metric(ABC):
    """Compares real and generated feeders."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def direction(self) -> Direction: ...

    @abstractmethod
    def compute(self, real: Sequence[Feeder],
                generated: Sequence[Feeder]) -> Mapping[str, float]:
        """Return one or more named scores.

        Returning a mapping lets a single computation report several related
        quantities (precision, recall, density, coverage) without repeating the
        expensive part.  Keys are suffixed with the metric name by the scorecard.
        """


class Descriptor(ABC):
    """Turns feeders into fixed-size vectors."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def __call__(self, feeders: Sequence[Feeder]) -> np.ndarray:
        """Return an ``[N, D]`` array, one row per feeder."""


class Distance(ABC):
    """Compares two sets of descriptor vectors."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def direction(self) -> Direction: ...

    @abstractmethod
    def __call__(self, real: np.ndarray, generated: np.ndarray) -> Mapping[str, float]:
        """Return named scores for the two ``[N, D]`` arrays."""


class DescriptorMetric(Metric):
    """A metric built from a descriptor and a distance.

    Because the two are independent, adding a descriptor makes it available to
    every distance already registered, and vice versa.
    """

    def __init__(self, descriptor: Descriptor, distance: Distance,
                 name: str | None = None):
        self.descriptor = descriptor
        self.distance = distance
        self._name = name or f"{descriptor.name}.{distance.name}"

    @property
    def name(self) -> str:
        return self._name

    @property
    def direction(self) -> Direction:
        return self.distance.direction

    def compute(self, real: Sequence[Feeder],
                generated: Sequence[Feeder]) -> Mapping[str, float]:
        return self.distance(self.descriptor(real), self.descriptor(generated))


class ReferenceMetric:
    """Wraps a metric with a real-versus-real reference value.

    Embedding-based scores have no absolute meaning — an FID of 30 is only
    interpretable next to the value two disjoint halves of the real data achieve.
    This computes that reference once so results can be reported against it
    rather than against 0 or 1.
    """

    def __init__(self, metric: Metric):
        self.metric = metric

    def reference(self, real: Sequence[Feeder], *, seed: int | None = 0
                  ) -> Mapping[str, float]:
        """Score one random half of the real feeders against the other."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(real))
        half = len(real) // 2
        if half == 0:
            raise ValueError("need at least two real feeders for a reference")
        a = [real[int(i)] for i in idx[:half]]
        b = [real[int(i)] for i in idx[half:2 * half]]
        return self.metric.compute(a, b)
