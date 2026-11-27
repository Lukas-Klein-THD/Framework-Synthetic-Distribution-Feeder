"""
Generating ampacity and phase for whole feeders with a conditional model.

A model only answers for single nodes: given the context fields of some nodes, it
draws an (ampacity, phase) for each.  But a node's context contains values of
other nodes (its parent, its previous sibling), so the nodes of a tree must be
generated in an order where those values already exist:

    1. the root
    2. for every node p, in node order: its children, one rank at a time
       (largest subtree first, as ranked by the encoder's TreeStructure)

Nodes are stored in BFS order, so every parent has a smaller index than its
children and is generated before them; within one parent, the ranks put each
child after the sibling before it.  :func:`walk` does this for a whole batch of
trees at once.

:class:`ConditionalAttributes` wraps the walk as a framework
:class:`~gridgen.generation.base.AttributeGenerator`, so it can be scored on real
topologies and combined with any topology model, exactly like the staged MDN.
Line length is not modelled here; it comes from a separate length source.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from gridgen.core.feeder import FeederSchema
from gridgen.generation.base import AttributeGenerator
from plugins.conditional.encoding import ContextEncoder
from plugins.conditional.models.base import ConditionalModel
from plugins.distribution.data import schema


def walk(encoder: ContextEncoder, model: ConditionalModel,
         adjacencies: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Generate every node of every tree top down; return the value codes [B, M]."""
    tree = encoder.structure(adjacencies, rng)           # parents, subtree bins, ranks
    value_code = np.full((tree.B, tree.m), -1)           # -1: not generated yet
    outcome = np.full((tree.B, tree.m), -1)

    def generate(due):
        """Sample the nodes marked in ``due`` [B, M] and store their values."""
        b, j = np.nonzero(due)
        if len(b) == 0:
            return
        fields = encoder.fields(tree, value_code, outcome, b, j)
        codes = model.sample(fields, rng)
        value_code[b, j] = codes
        outcome[b, j] = encoder.vocabulary.outcome_of_code(codes)

    generate(tree.parent == -1)                          # the roots
    for p in range(tree.m):                              # every node, as a possible parent
        for r in range(tree.max_rank + 1):               # its children, rank 0, 1, 2, ...
            generate((tree.parent == p) & (tree.rank == r))
    return value_code


class ConditionalAttributes(AttributeGenerator):
    """Ampacity and phase from a conditional model; line length from another generator.

    Parameters
    ----------
    model:
        A fitted :class:`ConditionalModel`; it carries its own encoder, so model and
        encoding always match.
    lengths:
        Any attribute generator that produces ``"length"`` (in practice the staged
        MDN); its other attributes are ignored.
    batch_size:
        Feeders generated per batch.
    """

    #: unconditional: the model has no category condition
    categories = ()

    def __init__(self, model: ConditionalModel, lengths: AttributeGenerator,
                 batch_size: int = 256):
        self.model = model
        self.lengths = lengths
        self.batch_size = batch_size

    @property
    def schema(self) -> FeederSchema:
        return schema()                                  # the same as the real data

    def generate(self, adjacencies, *, category=None, rng=None):
        """One ``{"length", "ampacity", "phase"}`` dict per adjacency.

        ``rng`` fixes everything: the sibling order, the sampling and, through a
        torch seed drawn from it, the lengths.
        """
        self._check_category(category)
        if rng is None:
            rng = np.random.default_rng()
        out = []
        for start in range(0, len(adjacencies), self.batch_size):
            chunk = adjacencies[start:start + self.batch_size]
            torch.manual_seed(int(rng.integers(2**31)))  # the length source samples with torch
            lengths = self.lengths.generate(chunk)
            codes = walk(self.model.encoder, self.model, chunk, rng)
            ampacity, phase = self.model.vocabulary.decode_code(codes)
            for i in range(len(chunk)):
                out.append({"length": np.asarray(lengths[i]["length"], dtype=np.float32),
                            "ampacity": ampacity[i],
                            "phase": phase[i]})
        return out
