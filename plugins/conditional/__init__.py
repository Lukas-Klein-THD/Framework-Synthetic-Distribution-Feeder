"""
Conditional ampacity-phase models: P(ampacity, phase | context), node by node.

Importing this package registers its components with the framework registry.

    from plugins import conditional
    from gridgen import build

    attributes = build("attribute_generator", "conditional",
                       checkpoint=".../conditional/t5.pt",
                       length_checkpoint=".../staged/attributes.pt")
    generator = build("generator", "staged-conditional",
                      topology_checkpoint=".../staged/topology.pt",
                      checkpoint=".../conditional/t5.pt",
                      length_checkpoint=".../staged/attributes.pt")

Any fitted model works as ``checkpoint``: the checkpoint names its own class.
Line length is not modelled here; it comes from the staged MDN.

Layout
------
``config``       context fields, table settings and the WP1 ladder
``encoding``     outcome vocabulary, tree structure, context fields
``models``       the conditional models (``table``) and ``load_model``
``generator``    the top-down walk and the framework attribute generator
``training``     fitting a model on a data source and saving its checkpoint
"""

from __future__ import annotations

from gridgen.core.registry import register
from gridgen.generation.staged import StagedGenerator

from plugins.conditional import config
from plugins.conditional.generator import ConditionalAttributes, walk
from plugins.conditional.models import load_model

__all__ = ["config", "ConditionalAttributes", "walk", "load_model",
           "build_conditional_attributes", "build_staged_conditional"]


@register("attribute_generator", "conditional")
def build_conditional_attributes(checkpoint: str, length_checkpoint: str,
                                 device: str = "cpu",
                                 batch_size: int = 256) -> ConditionalAttributes:
    """A fitted conditional model for ampacity and phase, the staged MDN for length."""
    from plugins.staged.attributes import MDNAttributes      # only needed for lengths

    lengths = MDNAttributes(checkpoint=length_checkpoint, device=device)
    return ConditionalAttributes(load_model(checkpoint), lengths, batch_size=batch_size)


@register("generator", "staged-conditional")
def build_staged_conditional(topology_checkpoint: str, checkpoint: str,
                             length_checkpoint: str, device: str = "cpu",
                             with_derived: bool = True) -> StagedGenerator:
    """The staged topology with conditional attributes in place of the MDN's.

    Only the attribute stage differs from ``build("generator", "staged")``, so the
    two can be compared directly.
    """
    from plugins.distribution.derived import TopologicalAttributes
    from plugins.staged.topology import TreeGANTopology

    topology = TreeGANTopology(checkpoint=topology_checkpoint, device=device)
    attributes = build_conditional_attributes(checkpoint, length_checkpoint, device=device)
    derived = [TopologicalAttributes()] if with_derived else []
    return StagedGenerator(topology=topology, attributes=attributes, derived=derived)
