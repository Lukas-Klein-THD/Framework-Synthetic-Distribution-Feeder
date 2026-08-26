"""
The data model everything in the framework exchanges.

A :class:`Feeder` is one distribution network: a graph plus named per-node
attributes plus metadata.  Every component — data sources, generators,
constraints, metrics — consumes and produces feeders, which is what lets them be
swapped independently.

Two deliberate choices shape this model:

* **Attributes are named, not positional.**  A feeder carries
  ``{"length": array, "phase": array, ...}`` rather than a fixed column order, so
  a model that generates a different attribute set plugs in without touching
  anything else.
* **Graph assumptions are declared, not implied.**  A :class:`GraphSpec` states
  whether a graph is directed, radial, rooted or fixed-size.  Radial and
  non-radial networks therefore coexist, and a component can check what it is
  given instead of assuming.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class AttributeKind(str, Enum):
    """How an attribute is distributed, which decides how it is modelled/scored."""

    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"


@dataclass(frozen=True)
class AttributeSpec:
    """Declaration of one per-node attribute.

    Parameters
    ----------
    name:
        Key under which the attribute appears in :attr:`Feeder.attributes`.
    kind:
        Continuous or categorical.  Metrics use this to pick a default distance
        (e.g. Wasserstein vs total variation) without being told.
    n_classes:
        Number of classes for categorical attributes.  ``None`` means the class
        set is learned from data rather than fixed.
    values:
        Optional explicit class values for a categorical attribute (e.g. the
        observed conductor ratings), in class order.
    unit:
        Physical unit, for documentation and export.
    description:
        Free text.
    """

    name: str
    kind: AttributeKind
    n_classes: int | None = None
    values: tuple[float, ...] | None = None
    unit: str | None = None
    description: str = ""

    @property
    def is_categorical(self) -> bool:
        return self.kind is AttributeKind.CATEGORICAL


@dataclass(frozen=True)
class GraphSpec:
    """What kind of graph a component expects or produces.

    ``radial=True`` means a connected, acyclic, single-rooted tree — the normal
    operating state of a distribution feeder.  ``radial=False`` admits meshed
    networks, so a generator that produces loops can declare that and be checked
    accordingly rather than silently violating an assumption.
    """

    directed: bool = True
    radial: bool = True
    root: int | None = 0
    num_nodes: int | None = None          # None = variable size

    def describe(self) -> str:
        parts = ["directed" if self.directed else "undirected",
                 "radial" if self.radial else "meshed"]
        if self.root is not None:
            parts.append(f"rooted at {self.root}")
        parts.append(f"{self.num_nodes} nodes" if self.num_nodes else "variable size")
        return ", ".join(parts)


@dataclass(frozen=True)
class FeederSchema:
    """The contract a set of feeders satisfies: graph shape plus attributes."""

    graph: GraphSpec = field(default_factory=GraphSpec)
    attributes: tuple[AttributeSpec, ...] = ()

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.attributes)

    def get(self, name: str) -> AttributeSpec:
        for a in self.attributes:
            if a.name == name:
                return a
        raise KeyError(f"{name!r} not in schema; have {self.names}")

    def subset(self, names: Sequence[str]) -> "FeederSchema":
        """A schema restricted to ``names`` (same graph spec)."""
        return FeederSchema(self.graph, tuple(self.get(n) for n in names))


@dataclass(frozen=True)
class Feeder:
    """One distribution network.

    Parameters
    ----------
    adjacency:
        ``[M, M]`` array.  For a directed radial graph ``adjacency[i, j] != 0``
        means *i* is the parent of *j*.  For an undirected graph it is symmetric.
    attributes:
        Mapping of attribute name to a ``[M]`` array, one value per node.
    metadata:
        Anything provenance-like: ``category``, ``source``, ``seed`` ...  Never
        interpreted by the framework except where documented.
    """

    adjacency: np.ndarray
    attributes: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # ── basic properties ─────────────────────────────────────────────────────
    @property
    def num_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @property
    def num_edges(self) -> int:
        a = np.asarray(self.adjacency)
        return int((a != 0).sum() // (1 if self.is_directed else 2))

    @property
    def is_directed(self) -> bool:
        a = np.asarray(self.adjacency)
        return not np.allclose(a, a.T)

    @property
    def category(self) -> str | None:
        return self.metadata.get("category")

    def parents(self) -> np.ndarray:
        """Parent index per node for a rooted graph (root maps to itself)."""
        return np.asarray(self.adjacency).argmax(axis=0)

    def with_attributes(self, **arrays: np.ndarray) -> "Feeder":
        """Copy with attributes added or replaced."""
        merged = dict(self.attributes)
        merged.update(arrays)
        return replace(self, attributes=merged)

    def with_metadata(self, **items: Any) -> "Feeder":
        merged = dict(self.metadata)
        merged.update(items)
        return replace(self, metadata=merged)

    def validate(self, schema: FeederSchema | None = None) -> list[str]:
        """Return a list of problems; empty means the feeder is well formed.

        Checks internal consistency always, and conformance to ``schema`` when
        one is given.  Returning problems rather than raising lets a caller
        decide whether a violation is fatal (a broken data source) or itself the
        measurement (a generator that breaks radiality).
        """
        problems: list[str] = []
        a = np.asarray(self.adjacency)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            problems.append(f"adjacency must be square, got {a.shape}")
            return problems
        m = a.shape[0]
        for name, arr in self.attributes.items():
            arr = np.asarray(arr)
            if arr.shape[0] != m:
                problems.append(f"attribute {name!r} has {arr.shape[0]} values for {m} nodes")
        if schema is None:
            return problems

        for spec in schema.attributes:
            if spec.name not in self.attributes:
                problems.append(f"missing attribute {spec.name!r}")
        g = schema.graph
        if g.num_nodes is not None and m != g.num_nodes:
            problems.append(f"expected {g.num_nodes} nodes, got {m}")
        if g.radial:
            problems.extend(_radiality_problems(a, root=g.root))
        return problems


def _radiality_problems(a: np.ndarray, root: int | None) -> list[str]:
    """Connected, acyclic, single-rooted, exactly M-1 edges."""
    problems: list[str] = []
    m = a.shape[0]
    binary = (a != 0)
    undirected = binary | binary.T
    np.fill_diagonal(undirected, False)
    n_edges = int(undirected.sum() // 2)
    if n_edges != m - 1:
        problems.append(f"radial graph needs {m - 1} edges, has {n_edges}")

    seen = np.zeros(m, dtype=bool)
    start = 0 if root is None else root
    stack, seen[start] = [start], True
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(undirected[u] & ~seen):
            seen[v] = True
            stack.append(v)
    if not seen.all():
        problems.append(f"graph is disconnected ({int((~seen).sum())} unreachable nodes)")

    if root is not None:
        in_degree = binary.sum(axis=0)
        if in_degree[root] != 0:
            problems.append(f"root {root} has {int(in_degree[root])} parents, expected 0")
        offenders = int((in_degree[np.arange(m) != root] != 1).sum())
        if offenders:
            problems.append(f"{offenders} non-root nodes do not have exactly one parent")
    return problems
