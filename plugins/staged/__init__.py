"""
Staged generator: topology first, then attributes, then derived quantities.

Importing this package registers its components with the framework registry.
It is written to be read as a template: each module implements exactly one
framework interface, and it depends on nothing outside this package except the
shared domain plugin :mod:`plugins.distribution` (data, constraints, metrics).

    from plugins import staged
    from gridgen import build

    generator = build("generator", "staged", device="cuda",
                      topology_checkpoint=..., attribute_checkpoint=...)

Layout
------
``config``       hyperparameters of both stages
``layers``       shared network blocks (directed GIN encoder)
``topology``     Stage 1 — adversarial topology model (radial by construction)
``profile``      feeder-level conditioning vector
``attributes``   Stage 2 — conditional mixture density model
``training``     training loops for both stages

Stage 3 (derived attributes) is model-independent and lives in
:mod:`plugins.distribution.derived`.
"""

from gridgen.core.registry import register
from gridgen.generation.staged import StagedGenerator

from plugins.distribution.derived import TopologicalAttributes
from plugins.staged import config
from plugins.staged.attributes import AttributeModel, MDNAttributes
from plugins.staged.profile import FeederProfile
from plugins.staged.topology import TreeGANTopology

__all__ = ["config", "TreeGANTopology", "MDNAttributes", "AttributeModel",
           "FeederProfile", "build_staged"]


@register("generator", "staged")
def build_staged(topology_checkpoint: str | None = None,
                 attribute_checkpoint: str | None = None,
                 device: str = "cpu",
                 constrain_phase: bool = False,
                 with_derived: bool = True) -> StagedGenerator:
    """Assemble the three stages as a single :class:`FeederGenerator`.

    ``StagedGenerator`` composes independent components, so the result is
    interchangeable with a model that generates everything at once.  Passing
    ``attribute_checkpoint=None`` yields a topology-only generator.
    """
    topology = TreeGANTopology(checkpoint=topology_checkpoint, device=device)
    attributes = None
    if attribute_checkpoint:
        attributes = MDNAttributes(checkpoint=attribute_checkpoint, device=device,
                                   constrain_phase=constrain_phase)
    derived = [TopologicalAttributes()] if with_derived else []
    return StagedGenerator(topology=topology, attributes=attributes, derived=derived)
