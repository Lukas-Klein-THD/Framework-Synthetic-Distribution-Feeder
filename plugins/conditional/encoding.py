"""
Encoding for the conditional ampacity-phase models.

Translates between the framework's physical attributes and the two discrete
quantities every conditional model works with:

    outcome   one index per (ampacity class, phase) pair observed in training.
              Ampacity and phase are sampled jointly, so their coupling is part
              of the outcome instead of something two separate heads must learn.
    context   named integer fields, each with a fixed number of values:

                  parent             the parent's (ampacity class, phase); root sentinel
                  subtree            subtree-size bin (node + descendants)
                  out_degree         the node's out-degree, capped (off by default)
                  sibling_subtree    subtree bin of the previous sibling, or of the
                                     next one for the first child; only-child sentinel
                  previous_sibling   outcome of the previous sibling; first-child and
                                     unseen-pair sentinels

              Which fields are used is set by :class:`~plugins.conditional.config.ContextConfig`.
              A model combines any prefix of the active fields into one index
              (:meth:`ContextEncoder.index`), which gives the table its coarser
              back-off contexts; the MLP and the QCBM use the fields directly.

The parent is encoded by its value (ampacity class, phase), not by its outcome
index.  For a joint model both carry the same information, but a factorised model
can generate a pair never seen in training, which has a value but no outcome.

Everything data-dependent (the ampacity classes and the observed pairs) is
fitted on the training split and stored with the checkpoint, so generation and
evaluation use exactly the training encoding.  The topology fields (subtree,
out-degree, sibling order) come from the adjacency alone, so they exist for real
and generated topologies alike.  The value fields (parent, previous sibling) come
from values already generated: attributes are produced top down, every node
after its parent and after the siblings before it.

The children of one parent are ordered largest subtree first, ties in random
order (:class:`TreeStructure`).  A node conditions on the sibling before it, so
for a node with two children the chain p(Y1 | ...) p(Y2 | ..., Y1) is exactly the
joint p(Y1, Y2 | parent, S1, S2).

The module is shared by every model (table, MLP, QCBM): the models differ only
in how they estimate P(outcome | context), never in what an outcome or a
context is.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gridgen.core.feeder import Feeder
from plugins.conditional.config import ContextConfig
from plugins.distribution.config import PHASE_LABELS
from plugins.distribution.derived import TopologicalAttributes

#: Number of phase codes
N_PHASES = len(PHASE_LABELS)
#: A rating further than this (in A) from every training class is an error, not a class.
AMP_TOLERANCE = 1.0
#: Context fields, in back-off order.
FIELDS = ("parent", "subtree", "out_degree", "sibling_subtree", "previous_sibling")


class OutcomeVocabulary:
    """Ampacity classes and the (ampacity class, phase) pairs seen in training.

    Pairs are stored as integer value codes ``ampacity_class * N_PHASES + phase``
    in sorted order; the outcome index of a pair is its position in that order.
    """

    def __init__(self, ampacity_classes: Sequence[float] | None = None,
                 pairs: Sequence[int] | None = None):
        self.ampacity_classes = None if ampacity_classes is None else \
            np.asarray(ampacity_classes, dtype=float)
        self.pairs = None if pairs is None else np.asarray(pairs, dtype=np.int64)

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit(self, feeders: Sequence[Feeder]) -> "OutcomeVocabulary":
        amps = np.concatenate([np.asarray(f.attributes["ampacity"], float) for f in feeders])
        self.ampacity_classes = np.unique(np.round(amps))
        codes = np.concatenate([self.value_code(f.attributes["ampacity"], f.attributes["phase"])
                                for f in feeders])
        self.pairs = np.unique(codes)
        return self

    def _check_fitted(self) -> None:
        if self.ampacity_classes is None or self.pairs is None:
            raise RuntimeError("OutcomeVocabulary is not fitted; call fit() or from_state()")

    # ── sizes ────────────────────────────────────────────────────────────────
    @property
    def n_ampacity(self) -> int:
        self._check_fitted()
        return len(self.ampacity_classes)

    @property
    def n_outcomes(self) -> int:
        self._check_fitted()
        return len(self.pairs)

    @property
    def n_value_codes(self) -> int:
        """Every (ampacity class, phase) combination, seen or not."""
        return self.n_ampacity * N_PHASES

    # ── physical values -> codes ─────────────────────────────────────────────
    def ampacity_class(self, ampacity) -> np.ndarray:
        """Index of the nearest ampacity class."""
        if self.ampacity_classes is None:
            raise RuntimeError("OutcomeVocabulary is not fitted; call fit() or from_state()")
        a = np.asarray(ampacity, dtype=float)
        dist = np.abs(a[..., None] - self.ampacity_classes)
        cls = dist.argmin(-1)
        if np.any(dist.min(-1) > AMP_TOLERANCE):
            bad = a[dist.min(-1) > AMP_TOLERANCE]
            raise ValueError(f"ampacity {bad[:5].tolist()} is not a training class "
                             f"(tolerance {AMP_TOLERANCE} A)")
        return cls

    @staticmethod
    def phase_code(phase) -> np.ndarray:
        p = np.asarray(phase, dtype=float)
        code = np.rint(p).astype(np.int64)
        if np.any(np.abs(p - code) > 1e-3) or np.any((code < 0) | (code >= N_PHASES)):
            raise ValueError(f"phase codes must be integers in 0..{N_PHASES - 1}")
        return code

    def value_code(self, ampacity, phase) -> np.ndarray:
        """``ampacity_class * N_PHASES + phase``; defined for every combination."""
        return self.ampacity_class(ampacity) * N_PHASES + self.phase_code(phase)

    def outcome_of_code(self, code) -> np.ndarray:
        """Outcome index of value codes; -1 for a pair never seen in training."""
        self._check_fitted()
        code = np.asarray(code, dtype=np.int64)
        i = np.clip(np.searchsorted(self.pairs, code), 0, len(self.pairs) - 1)
        return np.where(self.pairs[i] == code, i, -1)

    def encode(self, ampacity, phase) -> np.ndarray:
        """Outcome index per node; -1 for a pair never seen in training."""
        return self.outcome_of_code(self.value_code(ampacity, phase))

    # ── codes -> physical values ─────────────────────────────────────────────
    def decode(self, index) -> tuple[np.ndarray, np.ndarray]:
        """Ampacity and phase of outcome indices, as stored in ``Feeder.attributes``."""
        self._check_fitted()
        index = np.asarray(index, dtype=np.int64)
        if np.any((index < 0) | (index >= self.n_outcomes)):
            raise ValueError(f"outcome indices must be in 0..{self.n_outcomes - 1}")
        return self.decode_code(self.pairs[index])

    def decode_code(self, code) -> tuple[np.ndarray, np.ndarray]:
        """Ampacity and phase of value codes (seen in training or not)."""
        code = np.asarray(code, dtype=np.int64)
        ampacity = self.ampacity_classes[code // N_PHASES].astype(np.float32)
        return ampacity, (code % N_PHASES).astype(np.float32)

    # ── checkpoint ───────────────────────────────────────────────────────────
    def state(self) -> dict:
        self._check_fitted()
        return {"ampacity_classes": self.ampacity_classes.tolist(),
                "pairs": self.pairs.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "OutcomeVocabulary":
        return cls(state["ampacity_classes"], state["pairs"])


class TreeStructure:
    """The topology part of the context, for a batch of trees of equal size.

    Every array is ``[B, M]``.  Nodes must be in BFS order (every parent before
    its children), as in the data; generation relies on it.

    ``rank`` is a node's position among its siblings (largest subtree first, ties
    in random order), ``previous`` the sibling sampled just before it, and
    ``other`` the sibling whose subtree it conditions on: the previous one, or the
    next one for the first child.  -1 where there is none.
    """

    def __init__(self, adjacencies: Sequence[np.ndarray], config: ContextConfig,
                 rng: np.random.Generator):
        sizes = {np.asarray(a).shape for a in adjacencies}
        if len(sizes) != 1:
            raise ValueError(f"all trees in a batch must have the same size, got {sizes}")
        B, m = len(adjacencies), np.asarray(adjacencies[0]).shape[0]
        self.B, self.m = B, m
        shape = (B, m)
        self.parent = np.full(shape, -1, dtype=np.int64)
        self.subtree_size = np.ones(shape, dtype=np.int64)
        self.out_degree = np.zeros(shape, dtype=np.int64)
        self.rank = np.zeros(shape, dtype=np.int64)
        self.previous = np.full(shape, -1, dtype=np.int64)
        self.other = np.full(shape, -1, dtype=np.int64)
        derived = TopologicalAttributes()
        for b, adj in enumerate(adjacencies):
            adj = np.asarray(adj)
            parent = (adj != 0).argmax(axis=0)
            parent[0] = -1
            if np.any(parent[1:] >= np.arange(1, m)):
                raise ValueError(f"tree {b}: nodes are not in BFS order (a parent follows its child)")
            topo = derived.compute(Feeder(adj))
            self.parent[b] = parent
            self.subtree_size[b] = topo["n_descendants"].astype(np.int64) + 1
            self.out_degree[b] = topo["out_degree"].astype(np.int64)
            children = [[] for _ in range(m)]
            for j in range(1, m):
                children[parent[j]].append(j)
            for kids in children:
                if not kids:
                    continue
                kids = np.array(kids)
                kids = kids[np.lexsort((rng.random(len(kids)), -self.subtree_size[b, kids]))]
                self.rank[b, kids] = np.arange(len(kids))
                self.previous[b, kids[1:]] = kids[:-1]
                self.other[b, kids[1:]] = kids[:-1]
                if len(kids) > 1:
                    self.other[b, kids[0]] = kids[1]
        self.subtree_bin = np.digitize(self.subtree_size, config.subtree_edges)
        self.out_degree_bin = np.minimum(self.out_degree, config.max_out_degree)
        self.max_rank = int(self.rank.max())


class ContextEncoder:
    """Build the context fields of nodes, from the topology and the values so far."""

    def __init__(self, vocabulary: OutcomeVocabulary, config: ContextConfig | None = None):
        self.vocabulary = vocabulary
        self.config = config or ContextConfig()
        cfg = self.config
        use = {"parent": True, "subtree": cfg.use_subtree, "out_degree": cfg.use_out_degree,
               "sibling_subtree": cfg.use_sibling_subtree,
               "previous_sibling": cfg.use_previous_sibling}
        #: active fields, in back-off order
        self.active = tuple(name for name in FIELDS if use[name])
        # sentinels: one value past each field's regular range
        self.root = vocabulary.n_value_codes
        self.only_child = cfg.n_subtree_bins
        self.first_child = vocabulary.n_outcomes
        self.unseen = vocabulary.n_outcomes + 1
        #: number of values per field, sentinels included
        self.radix = {"parent": vocabulary.n_value_codes + 1,
                      "subtree": cfg.n_subtree_bins,
                      "out_degree": cfg.max_out_degree + 1,
                      "sibling_subtree": cfg.n_subtree_bins + 1,
                      "previous_sibling": vocabulary.n_outcomes + 2}

    def structure(self, adjacencies: Sequence[np.ndarray],
                  rng: np.random.Generator) -> TreeStructure:
        return TreeStructure(adjacencies, self.config, rng)

    def levels(self) -> list[tuple[str, ...]]:
        """Prefixes of the active fields, coarsest first: the back-off contexts."""
        return [self.active[:i] for i in range(1, len(self.active) + 1)]

    def fields(self, tree: TreeStructure, value_code: np.ndarray, outcome: np.ndarray,
               b: np.ndarray, j: np.ndarray) -> dict[str, np.ndarray]:
        """Active context fields of the nodes ``(b, j)``.

        ``value_code`` and ``outcome`` are ``[B, M]`` arrays of the values generated
        so far, -1 where a node has no value yet.  Every parent (and, if used, every
        previous sibling) of the requested nodes must already have one.
        """
        b, j = np.asarray(b), np.asarray(j)
        parent = tree.parent[b, j]
        root = parent < 0
        parent_code = value_code[b, np.maximum(parent, 0)]
        if np.any(~root & (parent_code < 0)):
            raise ValueError("a node's parent has no value yet; generate top down")
        out = {"parent": np.where(root, self.root, parent_code)}
        if "subtree" in self.active:
            out["subtree"] = tree.subtree_bin[b, j]
        if "out_degree" in self.active:
            out["out_degree"] = tree.out_degree_bin[b, j]
        if "sibling_subtree" in self.active:
            other = tree.other[b, j]
            out["sibling_subtree"] = np.where(other < 0, self.only_child,
                                              tree.subtree_bin[b, np.maximum(other, 0)])
        if "previous_sibling" in self.active:
            prev = tree.previous[b, j]
            first = prev < 0
            prev_code = value_code[b, np.maximum(prev, 0)]
            if np.any(~first & (prev_code < 0)):
                raise ValueError("a node's previous sibling has no value yet; "
                                 "generate siblings in rank order")
            prev_outcome = outcome[b, np.maximum(prev, 0)]
            out["previous_sibling"] = np.where(first, self.first_child,
                                               np.where(prev_outcome < 0, self.unseen, prev_outcome))
        return out

    def index(self, fields: dict[str, np.ndarray],
              names: Sequence[str] | None = None) -> np.ndarray:
        """Mixed-radix index over the named fields (default: all active ones)."""
        names = self.active if names is None else tuple(names)
        idx = np.zeros(np.shape(fields[names[0]]), dtype=np.int64)
        for name in names:
            x = np.asarray(fields[name], dtype=np.int64)
            if np.any((x < 0) | (x >= self.radix[name])):
                raise ValueError(f"context field {name!r} out of range 0..{self.radix[name] - 1}")
            idx = idx * self.radix[name] + x
        return idx

    def encode_feeders(self, feeders: Sequence[Feeder], rng: np.random.Generator
                       ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Context fields and true values of every node of real feeders (for fitting).

        Returns ``(fields, targets)``; ``targets`` holds per node the ``outcome``
        (-1 for an unseen pair), ``ampacity_class``, ``phase`` and ``feeder`` index.
        """
        voc = self.vocabulary
        tree = self.structure([np.asarray(f.adjacency) for f in feeders], rng)
        amp = np.stack([voc.ampacity_class(f.attributes["ampacity"]) for f in feeders])
        phase = np.stack([voc.phase_code(f.attributes["phase"]) for f in feeders])
        code = amp * N_PHASES + phase
        outcome = voc.outcome_of_code(code)
        b, j = (a.ravel() for a in np.indices((tree.B, tree.m)))
        fields = self.fields(tree, code, outcome, b, j)
        targets = {"outcome": outcome[b, j], "ampacity_class": amp[b, j],
                   "phase": phase[b, j], "feeder": b}
        return fields, targets
