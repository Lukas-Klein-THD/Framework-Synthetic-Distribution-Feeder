"""
Settings of the conditional ampacity-phase models.

Data location, graph size and evaluation settings are shared by every model and
live in :mod:`plugins.distribution.config`; this module holds only what is
specific to the conditional models.  Each switch in :class:`ContextConfig` adds
one field to the context, so the models of a comparison differ in exactly one
setting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from plugins.distribution.config import CHECKPOINT_ROOT

#: Where this plugin's fitted models are written.
CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, "conditional")


@dataclass(frozen=True)
class ContextConfig:
    """Which fields make up a node's context, beyond the parent's value.

    The fields are used in this order: parent, subtree, out-degree, sibling
    subtree, previous sibling.  Models that smooth over coarser contexts (the
    table) back off along this order.
    """

    #: subtree-size bin of the node (node + descendants)
    use_subtree: bool = True
    #: the node's own out-degree, capped; carries little beyond the subtree size
    use_out_degree: bool = False
    #: subtree bin of a sibling: the previous one, or the next one for the first child
    use_sibling_subtree: bool = True
    #: outcome of the previous sibling in the sampling order
    use_previous_sibling: bool = True
    #: bin edges on the subtree size: 1, 2, 3-4, 5-8, 9-16, 17-32, >=33
    subtree_edges: tuple[int, ...] = (2, 3, 5, 9, 17, 33)
    #: out-degrees above this share the last bin
    max_out_degree: int = 3

    @property
    def n_subtree_bins(self) -> int:
        return len(self.subtree_edges) + 1


@dataclass(frozen=True)
class TableConfig:
    """Hierarchical back-off table."""
    #: Dirichlet strength towards the next coarser context (held-out bits stable for 1-20)
    beta: float = 5.0
    #: p(A | ctx) p(P | ctx) instead of p(Y | ctx): the factorised control (T1)
    factorised: bool = False


#: the WP1 ladder: each rung changes one setting against the one before
LADDER = {
    "t1": (ContextConfig(use_subtree=False, use_sibling_subtree=False,
                         use_previous_sibling=False), TableConfig(factorised=True)),
    "t2": (ContextConfig(use_subtree=False, use_sibling_subtree=False,
                         use_previous_sibling=False), TableConfig()),
    "t3": (ContextConfig(use_sibling_subtree=False, use_previous_sibling=False), TableConfig()),
    "t4": (ContextConfig(use_previous_sibling=False), TableConfig()),
    "t5": (ContextConfig(), TableConfig()),
}


@dataclass(frozen=True)
class MLPConfig:
    """
    Feedforward network for the conditional model.

    Each active context field is looked up in its own embedding; the embeddings are
    concatenated (n_fields * embedding_dim inputs) and mapped through ``n_layers``
    hidden layers to one logit per outcome.  Trained by cross-entropy with early
    stopping on held-back training feeders.
    """
    # ── architecture ──
    #: numbers per field value in each embedding
    embedding_dim: int = 16
    #: width of every hidden layer
    hidden_dim: int = 128
    #: number of hidden layers (0: the embeddings feed the output layer directly)
    n_layers: int = 2
    #: share of hidden units switched off at random during training
    dropout: float = 0.1

    # ── optimisation ──
    #: Adam learning rate
    lr: float = 1e-3
    #: L2 penalty on the weights
    weight_decay: float = 0.0
    #: rows per gradient step
    batch_size: int = 512
    #: upper bound on the epochs; early stopping normally ends training sooner
    max_epochs: int = 100

    # ── early stopping ──
    #: share of the training feeders held back for early stopping (split by feeder)
    validation_fraction: float = 0.1
    #: epochs without a better validation loss before training stops
    patience: int = 5

    #: weight initialisation, validation split, shuffling and dropout
    seed: int = 0