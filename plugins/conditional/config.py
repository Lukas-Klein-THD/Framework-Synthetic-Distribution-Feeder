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
