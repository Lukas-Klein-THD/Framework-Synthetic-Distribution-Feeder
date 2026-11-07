"""
Encoding for the conditional ampacity-phase models.

Translates between the framework's physical attributes and the two discrete
quantities every conditional model works with:

    outcome   one index per (ampacity, phase) pair observed in training.
              Ampacity and phase are sampled jointly, so their coupling is part
              of the outcome instead of something two separate heads must learn.
    context   one index per (parent outcome, out-degree bin, subtree-size bin).
              The root has no parent and uses a sentinel parent outcome.

Everything data-dependent (the ampacity classes and the observed pairs) is
fitted on the training split and stored with the checkpoint, so generation and
evaluation use exactly the training encoding.  Context features are computed
from the adjacency alone, so they exist for real and generated topologies alike.

The module is shared by every model (table, MLP, QCBM): the models differ only
in how they estimate P(outcome | context), never in what an outcome or a
context is.
"""

from __future__ import annotations

import numpy as np

from gridgen.core.feeder import Feeder
from plugins.distribution.config import PHASE_LABELS
from plugins.distribution.derived import TopologicalAttributes

# Number of phase codes
N_PHASES = len(PHASE_LABELS)
AMP_TOLERANCE = 1.0
SUBTREE_EDGES = (2, 3, 5, 9, 17, 33)
MAX_OUT_DEGREE = 3


class OutcomeVocabulary:
    """Map observed (ampacity, phase) pairs to integer labels."""

    def __init__(self):
        self.ampacity_classes = None
        self.pairs = None
        self.pair_to_index = None

    def fit(self, feeders):
        # 1. Collect ampacity and phase values
        combined_values = []

        for feeder in feeders:
            ampacity = np.round(feeder.attributes["ampacity"])
            phase = feeder.attributes["phase"]

            combined_values.extend(zip(ampacity, phase))

        self.pairs = sorted(set(combined_values))

        self.pair_to_index = {
            pair: index
            for index, pair in enumerate(self.pairs)
        }

    def encode(self, pair):
        # Convert attribute pairs to outcome indices
        return self.pair_to_index.get(pair, -1)

    def decode(self, index):
        # Convert outcome indices back to physical attributes
        return self.pairs[index]


class ContextEncoder:
    """Build conditioning information for each node."""

    def __init__(self, n_outcomes, use_degree=False, use_subtree=False):
        self.n_outcomes = n_outcomes
        self.root_outcome = n_outcomes
        self.use_degree = use_degree
        self.use_subtree = use_subtree

    def node_features(self, adjacency):
        # Determine parent, out-degree and subtree size
        
        raise NotImplementedError

    def encode_context(self, parent_outcome, degree_bin=0, subtree_bin=0):
        # Convert conditioning information to one context index
        raise NotImplementedError
