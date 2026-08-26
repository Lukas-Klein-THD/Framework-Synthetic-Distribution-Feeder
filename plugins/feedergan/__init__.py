"""
Reference plugin: a staged feeder generator and its evaluation.

Importing this package registers every component with the framework registry.
It is written to be read as a template — each module implements exactly one
framework interface, and nothing here depends on code outside this package.

    from plugins import feedergan
    from gridgen import build, available

    available("generator")          # -> ['staged-feedergan']
    generator = build("generator", "staged-feedergan", device="cuda")

Layout
------
``config``       paths and hyperparameters, all in one place
``data``         :class:`DataSource` over preprocessed tensors, category labeller
``layers``       shared network blocks (directed GIN encoder)
``topology``     Stage 1 — adversarial topology model
``profile``      feeder-level conditioning vector
``attributes``   Stage 2 — conditional mixture density model
``derived``      Stage 3 — attributes computed, not learned
``constraints``  phase-subset (hard) and ampacity-tapering (soft)
``metrics``      descriptors, distances and two ready-made suites
``powerflow``    OpenDSS snapshot solve as a metric
``training``     training loops for both stages
"""

from gridgen.core.registry import register
from gridgen.generation.staged import StagedGenerator

from plugins.feedergan import config
from plugins.feedergan.attributes import AttributeModel, MDNAttributes
from plugins.feedergan.constraints import AmpacityTapering, PhaseSubset
from plugins.feedergan.data import SourcePathLabeler, TensorFeederSource, schema
from plugins.feedergan.derived import TopologicalAttributes
from plugins.feedergan.metrics import attributed_metrics, structural_metrics
from plugins.feedergan.profile import FeederProfile
from plugins.feedergan.topology import TreeGANTopology

__all__ = [
    "config", "schema",
    "TensorFeederSource", "SourcePathLabeler",
    "TreeGANTopology", "MDNAttributes", "AttributeModel", "FeederProfile",
    "TopologicalAttributes", "PhaseSubset", "AmpacityTapering",
    "structural_metrics", "attributed_metrics",
    "build_staged",
]


@register("generator", "staged-feedergan")
def build_staged(topology_checkpoint: str | None = None,
                 attribute_checkpoint: str | None = None,
                 device: str = "cpu",
                 constrain_phase: bool = False,
                 with_derived: bool = True) -> StagedGenerator:
    """Assemble the reference pipeline as a single :class:`FeederGenerator`.

    The three stages are independent components; ``StagedGenerator`` composes
    them so the result is interchangeable with a model that generates everything
    at once.  Passing ``attribute_checkpoint=None`` yields a topology-only
    generator, which is how a new topology model is evaluated before any
    attribute model exists.
    """
    topology = TreeGANTopology(checkpoint=topology_checkpoint, device=device)
    attributes = None
    if attribute_checkpoint:
        attributes = MDNAttributes(checkpoint=attribute_checkpoint, device=device,
                                   constrain_phase=constrain_phase)
    derived = [TopologicalAttributes()] if with_derived else []
    return StagedGenerator(topology=topology, attributes=attributes, derived=derived)
