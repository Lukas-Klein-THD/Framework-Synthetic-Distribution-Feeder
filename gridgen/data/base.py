"""
Data-source interfaces.

A :class:`DataSource` is a sequence of real feeders plus the schema they satisfy.
Keeping loading behind an interface means a new dataset — a different country, a
meshed network, a different file format — is a plugin rather than an edit.

Category labels are separated into :class:`CategoryLabeler` because how a feeder
is classified (folder layout, naming convention, a clustering of its attributes)
is independent of how it is loaded.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterator, Mapping, Sequence

from gridgen.core.feeder import Feeder, FeederSchema


class DataSource(ABC):
    """A collection of real feeders."""

    @property
    @abstractmethod
    def schema(self) -> FeederSchema:
        """Graph shape and attributes the loaded feeders satisfy."""

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> Feeder: ...

    def __iter__(self) -> Iterator[Feeder]:
        for i in range(len(self)):
            yield self[i]

    # ── convenience shared by every source ───────────────────────────────────
    def sample(self, n: int, *, seed: int | None = None) -> list[Feeder]:
        """A random subset of ``n`` feeders (without replacement)."""
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self))[:n]
        return [self[int(i)] for i in idx]

    def by_category(self) -> Mapping[str, list[int]]:
        """Indices grouped by ``metadata['category']``; unlabelled under ``None``."""
        groups: dict[str, list[int]] = defaultdict(list)
        for i in range(len(self)):
            groups[self[i].category].append(i)  # type: ignore[index]
        return dict(groups)

    def filter(self, predicate) -> "SubsetSource":
        """A view containing only feeders satisfying ``predicate(feeder)``."""
        keep = [i for i in range(len(self)) if predicate(self[i])]
        return SubsetSource(self, keep)

    def with_category(self, category: str) -> "SubsetSource":
        return self.filter(lambda f: f.category == category)


class SubsetSource(DataSource):
    """A lazy view over a subset of another source."""

    def __init__(self, source: DataSource, indices: Sequence[int]):
        self._source = source
        self._indices = list(indices)

    @property
    def schema(self) -> FeederSchema:
        return self._source.schema

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: int) -> Feeder:
        return self._source[self._indices[index]]

    def __repr__(self) -> str:
        return f"SubsetSource({type(self._source).__name__}, n={len(self)})"


class CategoryLabeler(ABC):
    """Assigns a category to a feeder.

    Separate from loading because classification schemes differ from datasets:
    one corpus encodes the category in its directory layout, another in a naming
    convention, a third might cluster feeders by their attributes.
    """

    @property
    @abstractmethod
    def categories(self) -> tuple[str, ...]:
        """Every label this labeler can emit."""

    @abstractmethod
    def label(self, feeder: Feeder) -> str | None:
        """Category for one feeder, or ``None`` if it cannot be determined."""

    def apply(self, feeder: Feeder) -> Feeder:
        """Return the feeder with ``metadata['category']`` filled in."""
        return feeder.with_metadata(category=self.label(feeder))
