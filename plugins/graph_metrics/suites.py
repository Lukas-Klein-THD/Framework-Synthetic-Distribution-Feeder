"""
Ready-made metric suites: a structural one and an attributed one.

The attributed suite takes the node encoder as an argument; the domain plugin
supplies it (``plugins.distribution.AttributeEncoder`` for feeders).
"""

from __future__ import annotations

from gridgen.evaluation.base import DescriptorMetric

from plugins.graph_metrics.descriptors import (
    DegreeHistogram, LaplacianSpectrum, RandomGINEmbedding)
from plugins.graph_metrics.distances import (
    FrechetDistance, MaximumMeanDiscrepancy, PrecisionRecallDensityCoverage)


def structural_metrics(max_degree: int = 24, seed: int = 0) -> list[DescriptorMetric]:
    """Topology-only metrics."""
    gin = RandomGINEmbedding(seed=seed)
    return [
        DescriptorMetric(DegreeHistogram(max_degree), MaximumMeanDiscrepancy()),
        DescriptorMetric(LaplacianSpectrum(), MaximumMeanDiscrepancy()),
        DescriptorMetric(gin, FrechetDistance()),
        DescriptorMetric(gin, PrecisionRecallDensityCoverage()),
    ]


def attributed_metrics(encoder, seed: int = 0) -> list[DescriptorMetric]:
    """Metrics over the joint of structure and attributes.

    ``encoder`` maps a feeder to per-node vectors, normally a domain encoder
    fitted on the real feeders.
    """
    gin = RandomGINEmbedding(seed=seed, encoder=encoder)
    return [
        DescriptorMetric(gin, FrechetDistance()),
        DescriptorMetric(gin, PrecisionRecallDensityCoverage()),
    ]
