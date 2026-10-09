"""
Data source for preprocessed feeder tensors.

Reads ``.pt`` files containing a segment-as-node feeder and presents them as
:class:`~gridgen.core.feeder.Feeder` objects.  This is the only module that knows
the on-disk format; everything downstream sees the framework's data model.

Every node is a line segment: switching and protection devices are merged into
the segments they connect, and lengths are true km.

Each stored graph is a binary, BFS-ordered, upper-triangular, in-degree-one tree
rooted at node 0.  Because BFS ordering guarantees that a parent precedes its
children, the first ``M`` nodes are closed under the parent relation and form a
valid connected rooted subtree — which is what makes fixed-size truncation
structurally meaningful rather than an arbitrary crop.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Sequence

import numpy as np

from gridgen.core.feeder import (
    AttributeKind, AttributeSpec, Feeder, FeederSchema, GraphSpec)
from gridgen.core.registry import register
from gridgen.data.base import CategoryLabeler, DataSource

from plugins.distribution import config

#: Organic attributes every model generates, in a fixed order.
ATTRIBUTES = (
    AttributeSpec("length", AttributeKind.CONTINUOUS, unit="km",
                  description="segment length"),
    AttributeSpec("ampacity", AttributeKind.CATEGORICAL,
                  description="conductor current rating (observed class set)"),
    AttributeSpec("phase", AttributeKind.CATEGORICAL,
                  n_classes=len(config.PHASE_LABELS),
                  values=tuple(float(i) for i in range(len(config.PHASE_LABELS))),
                  description="phase configuration a..abc"),
)


def schema(num_nodes: int = config.NUM_NODES) -> FeederSchema:
    """The schema real and generated feeders both satisfy."""
    return FeederSchema(GraphSpec(directed=True, radial=True, root=0,
                                  num_nodes=num_nodes), ATTRIBUTES)


@register("data_source", "feeder-tensors")
class TensorFeederSource(DataSource):
    """Feeders loaded from preprocessed ``.pt`` tensors.

    Parameters
    ----------
    directory:
        Searched recursively for ``.pt`` files.
    num_nodes:
        Fixed size ``M``.  Graphs with fewer nodes are dropped, larger ones are
        truncated to their first ``M`` BFS nodes.
    include_decomposed:
        Whether to include sub-feeders extracted from larger networks (files
        under a ``subfeeders`` directory).  They enlarge the training set but
        inherit their parent's category, so analyses that group by category
        normally exclude them.
    labeler:
        Optional :class:`CategoryLabeler` applied on load.
    cache:
        Keep loaded feeders in memory.  Training touches every graph many
        times; at M = 128 the full set is about 170 MB.
    extras:
        Additional stored quantities to load as attributes, for models that
        learn them: ``"distance"`` (km from the source), ``"pseudo_load"``
        (downstream transformer kVA, over the full feeder) and ``"level"``
        (FeederGAN level class).  They are not part of the schema; evaluation
        uses only the organic attributes.
    """

    EXTRAS = ("distance", "pseudo_load", "level")

    def __init__(self, directory: str | None = None,
                 num_nodes: int = config.NUM_NODES,
                 include_decomposed: bool = True,
                 labeler: "CategoryLabeler | None" = None,
                 max_graphs: int | None = None,
                 cache: bool = True,
                 extras: Sequence[str] = ()):
        unknown = set(extras) - set(self.EXTRAS)
        if unknown:
            raise ValueError(f"unknown extras {sorted(unknown)}; choose from {self.EXTRAS}")
        self.extras = tuple(extras)
        self.directory = os.path.abspath(directory or config.DATA_DIR)
        self.num_nodes = num_nodes
        self.labeler = labeler
        self._cache: dict[int, Feeder] | None = {} if cache else None
        if not os.path.isdir(self.directory):
            raise FileNotFoundError(
                f"data directory not found: {self.directory}. Set GRIDGEN_DATA_DIR "
                "or pass directory=...")

        paths: list[str] = []
        for root, _, files in os.walk(self.directory):
            if not include_decomposed and os.path.basename(root) == "subfeeders":
                continue
            paths.extend(os.path.join(root, f) for f in files if f.endswith(".pt"))
        self._paths = [p for p in sorted(paths) if self._size(p) >= num_nodes]
        if max_graphs:
            self._paths = self._paths[:max_graphs]
        if not self._paths:
            raise RuntimeError(
                f"no graphs with >= {num_nodes} nodes under {self.directory}")

    # ── DataSource ───────────────────────────────────────────────────────────
    @property
    def schema(self) -> FeederSchema:
        return schema(self.num_nodes)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, index: int) -> Feeder:
        if self._cache is not None and index in self._cache:
            return self._cache[index]
        feeder = self._load(index)
        if self._cache is not None:
            self._cache[index] = feeder
        return feeder

    def _load(self, index: int) -> Feeder:
        import torch
        blob = torch.load(self._paths[index], map_location="cpu", weights_only=False)
        tensors, meta = blob["tensors"], blob["meta"]
        m = self.num_nodes

        adjacency = tensors["A_raw"].numpy()[:m, :m].astype(np.float32)
        numeric = tensors["X_num_raw"].numpy()[:m]          # km, A, km, kVA
        attributes = {
            "length": numeric[:, 0].astype(np.float32),
            "ampacity": numeric[:, 1].astype(np.float32),
            "phase": tensors["phase_labels"].numpy()[:m].astype(np.float32),
        }
        stored = {"distance": numeric[:, 2], "pseudo_load": numeric[:, 3],
                  "level": tensors["level_labels"].numpy()[:m]}
        for name in self.extras:
            attributes[name] = stored[name].astype(np.float32)
        feeder = Feeder(adjacency, attributes,
                        {"source": meta.get("source_file"), "path": self._paths[index]})
        return self.labeler.apply(feeder) if self.labeler else feeder

    # ── size cache ───────────────────────────────────────────────────────────
    @staticmethod
    @lru_cache(maxsize=None)
    def _size(path: str) -> int:
        import torch
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            return int(blob["tensors"]["A_raw"].shape[0])
        except Exception:
            return -1

    def __repr__(self) -> str:
        return f"TensorFeederSource(n={len(self)}, M={self.num_nodes})"


@register("category_labeler", "source-path")
class SourcePathLabeler(CategoryLabeler):
    """Recovers the network category from the provenance path.

    Two conventions are supported: an explicit category directory
    (``.../rural/...``) and the SMART-DS region naming ``P<n>U`` / ``P<n>R``,
    where the suffix denotes an urban or rural region.
    """

    CATEGORIES = ("industrial", "rural", "urban-suburban")
    _REGION = re.compile(r"^p\d+([ur])$")

    @property
    def categories(self) -> tuple[str, ...]:
        return self.CATEGORIES

    def label(self, feeder: Feeder) -> str | None:
        source = str(feeder.metadata.get("source", ""))
        parts = [p.lower() for p in source.replace("\\", "/").split("/")]
        for part in parts:
            if part in self.CATEGORIES:
                return part
        for part in parts:
            m = self._REGION.match(part)
            if m:
                return "urban-suburban" if m.group(1) == "u" else "rural"
        return None
