"""
Generic graph metrics: descriptors, distances and two ready-made suites.

Nothing here knows about distribution feeders or about any generative model.
Descriptors turn graphs (optionally with per-node vectors from an encoder) into
vectors; distances compare two sets of vectors.  Keeping them apart means a new
descriptor works with every distance and vice versa — adding an embedding does
not require reimplementing MMD.

Three points of interpretation are built in:

* Embedding scores have **no absolute meaning**; read them against the
  real-versus-real reference the evaluator computes.
* **Precision and recall must be read together.**  A model producing a narrow
  set of plausible graphs scores high precision and near-zero recall.
* MMD uses the **maximum over a bandwidth range** rather than a single kernel
  width, since one bandwidth can hide a difference the next one exposes.

Domain-specific evaluation (the attribute encoder, constraints, power flow and
the scorecard) lives in the domain plugin, :mod:`plugins.distribution`.

Layout
------
``descriptors``  degree histogram, Laplacian spectrum, random-GIN embedding
``distances``    MMD, Frechet distance (FD), precision/recall/density/coverage
``suites``       ``structural_metrics`` and ``attributed_metrics``
``graphs``       undirected view, radial-tree check
"""

from plugins.graph_metrics.descriptors import (
    DegreeHistogram, LaplacianSpectrum, RandomGINEmbedding)
from plugins.graph_metrics.distances import (
    FrechetDistance, MaximumMeanDiscrepancy, PrecisionRecallDensityCoverage)
from plugins.graph_metrics.graphs import is_radial_tree, undirected
from plugins.graph_metrics.suites import attributed_metrics, structural_metrics

__all__ = [
    "DegreeHistogram", "LaplacianSpectrum", "RandomGINEmbedding",
    "MaximumMeanDiscrepancy", "FrechetDistance", "PrecisionRecallDensityCoverage",
    "structural_metrics", "attributed_metrics",
    "is_radial_tree", "undirected",
]
