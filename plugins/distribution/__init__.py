"""
Distribution-feeder domain plugin: data, constraints and evaluation.

Everything here is independent of any generative model.  Model plugins
(``plugins.staged``, ``plugins.feedergan_improved``) depend on this package;
it depends on none of them.  That is what lets two very different models be
trained on the same data and scored by the same harness.  The generic graph
metrics it builds on live in :mod:`plugins.graph_metrics`.

Layout
------
``config``       data location, fixed graph size, power-flow operating point
``data``         :class:`DataSource` over preprocessed tensors, category labeller
``derived``      attributes computed from a finished feeder (depth, distance, ...)
``constraints``  phase subset (hard) and ampacity tapering (soft)
``metrics``      attribute encoder and the three scorecard scores
``powerflow``    OpenDSS snapshot solve as a metric (needs ``dss-python``)
``reconstruction``  soft adjacency -> binary graph (FeederGAN Algorithm 2)
"""

from plugins.distribution import config
from plugins.distribution.constraints import AmpacityTapering, PhaseSubset
from plugins.distribution.data import (
    ATTRIBUTES, SourcePathLabeler, TensorFeederSource, schema)
from plugins.distribution.derived import TopologicalAttributes
from plugins.distribution.metrics import AttributeEncoder
# generic suites, re-exported for compatibility (defined in plugins.graph_metrics)
from plugins.graph_metrics import attributed_metrics, structural_metrics  # noqa: F401

__all__ = [
    "config", "schema", "ATTRIBUTES",
    "TensorFeederSource", "SourcePathLabeler",
    "TopologicalAttributes", "PhaseSubset", "AmpacityTapering",
    "AttributeEncoder",
]
