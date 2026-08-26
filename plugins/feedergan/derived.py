"""
Stage 3 — attributes computed rather than learned.

Depth, distance to the source and downstream device counts are exact functions of
the graph and the generated attributes.  Learning them would invite
inconsistency: a model could emit a distance that disagrees with the lengths it
also emitted.  Computing them keeps the feeder internally consistent by
construction, and removes four quantities from the learning problem.
"""

from __future__ import annotations

import numpy as np

from gridgen.core.feeder import Feeder
from gridgen.core.registry import register
from gridgen.generation.base import DerivedAttributes


@register("derived_attributes", "feeder-topology")
class TopologicalAttributes(DerivedAttributes):
    """Depth, cumulative distance, degrees and downstream counts."""

    @property
    def produces(self) -> tuple[str, ...]:
        return ("depth", "distance", "out_degree", "n_descendants")

    def compute(self, feeder: Feeder) -> dict[str, np.ndarray]:
        adjacency = np.asarray(feeder.adjacency)
        m = adjacency.shape[0]
        parents = adjacency.argmax(axis=0)
        length = np.asarray(feeder.attributes.get("length", np.zeros(m)), dtype=float)

        depth = np.zeros(m)
        distance = np.zeros(m)
        for j in range(1, m):                       # BFS order: parent already done
            p = int(parents[j])
            depth[j] = depth[p] + 1.0
            distance[j] = distance[p] + length[j]

        descendants = np.zeros(m)
        for j in range(m - 1, 0, -1):               # reverse order: children first
            descendants[int(parents[j])] += descendants[j] + 1.0

        return {
            "depth": depth,
            "distance": distance,
            "out_degree": (adjacency != 0).sum(axis=1).astype(float),
            "n_descendants": descendants,
        }
