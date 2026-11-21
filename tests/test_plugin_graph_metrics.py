"""
Generic graph-metrics plugin: domain independence, compatibility re-exports and
basic metric behaviour.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

from gridgen import Feeder
from gridgen.evaluation.base import Direction

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(n):
    a = np.zeros((n, n), dtype=np.float32)
    for j in range(1, n):
        a[j - 1, j] = 1.0
    return a


def _star(n):
    a = np.zeros((n, n), dtype=np.float32)
    a[0, 1:] = 1.0
    return a


def test_graph_metrics_does_not_import_the_domain_plugin():
    code = ("import sys, plugins.graph_metrics; "
            "assert not any(m.startswith('plugins.distribution') for m in sys.modules)")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=REPO)


def test_distribution_re_exports_are_the_same_objects():
    import plugins.distribution as dist
    import plugins.distribution.metrics as old
    import plugins.graph_metrics as new
    for name in ("DegreeHistogram", "LaplacianSpectrum", "RandomGINEmbedding",
                 "MaximumMeanDiscrepancy", "FrechetDistance",
                 "PrecisionRecallDensityCoverage", "structural_metrics",
                 "attributed_metrics", "is_radial_tree"):
        assert getattr(old, name) is getattr(new, name)
    assert dist.structural_metrics is new.structural_metrics


def test_distances_separate_different_graph_families():
    from plugins.graph_metrics import (
        DegreeHistogram, MaximumMeanDiscrepancy, PrecisionRecallDensityCoverage)
    paths = [Feeder(_path(n)) for n in range(8, 20)]
    stars = [Feeder(_star(n)) for n in range(8, 20)]
    d = DegreeHistogram(max_degree=24)
    mmd = MaximumMeanDiscrepancy()
    assert mmd.direction is Direction.LOWER_IS_BETTER
    same = mmd(d(paths[::2]), d(paths[1::2]))["value"]
    different = mmd(d(paths), d(stars))["value"]
    assert different > same
    prdc = PrecisionRecallDensityCoverage(nearest_k=3)
    assert prdc(d(paths), d(paths))["precision"] == pytest.approx(1.0)
    assert prdc(d(paths), d(stars))["precision"] < 0.5


def test_suites_have_the_documented_composition():
    from plugins.graph_metrics import attributed_metrics, structural_metrics
    assert len(structural_metrics()) == 4
    assert len(attributed_metrics(encoder=lambda f: np.ones((len(f.adjacency), 2)))) == 2
