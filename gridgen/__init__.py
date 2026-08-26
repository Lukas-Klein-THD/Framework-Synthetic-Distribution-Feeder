"""
gridgen — a pluggable framework for generating and evaluating synthetic
distribution networks.

The package defines *interfaces only*.  Models, datasets, constraints and metrics
live in plugin packages that register themselves; nothing here imports torch or
any modelling library.

Read :mod:`gridgen.core.feeder` first — the :class:`Feeder` data model is the
contract every other component is written against.
"""

from gridgen.core.feeder import (
    AttributeKind, AttributeSpec, Feeder, FeederSchema, GraphSpec)
from gridgen.core.registry import available, build, catalogue, register
from gridgen.constraints.base import Constraint, ValidationResult
from gridgen.data.base import CategoryLabeler, DataSource, SubsetSource
from gridgen.evaluation.base import (
    Descriptor, DescriptorMetric, Direction, Distance, Metric, ReferenceMetric)
from gridgen.evaluation.scorecard import Evaluator, Scorecard
from gridgen.generation.base import (
    AttributeGenerator, DerivedAttributes, FeederGenerator, Generator,
    TopologyGenerator)
from gridgen.generation.staged import StagedGenerator
from gridgen.pipeline import Pipeline, Report

__version__ = "0.1.0"

__all__ = [
    "Feeder", "FeederSchema", "GraphSpec", "AttributeSpec", "AttributeKind",
    "DataSource", "SubsetSource", "CategoryLabeler",
    "Generator", "FeederGenerator", "TopologyGenerator", "AttributeGenerator",
    "DerivedAttributes", "StagedGenerator",
    "Constraint", "ValidationResult",
    "Metric", "Descriptor", "Distance", "DescriptorMetric", "Direction",
    "ReferenceMetric", "Evaluator", "Scorecard",
    "Pipeline", "Report",
    "register", "build", "available", "catalogue",
]
