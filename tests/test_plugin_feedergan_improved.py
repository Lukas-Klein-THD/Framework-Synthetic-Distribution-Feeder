"""
Tests for the improved FeederGAN plugin.

The model is untrained here; the tests check contracts: it is a joint
FeederGenerator, its reconstruction and relabelling are correct, and its
attribute encoding round-trips.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gridgen import Feeder, FeederGenerator, available
from plugins import feedergan_improved  # noqa: F401  (registration)
from plugins.distribution.metrics import is_radial_tree
from plugins.feedergan_improved import FeederGANImproved
from plugins.feedergan_improved.encoding import AMPACITIES, CAT_VOCAB_SIZES, decode, encode
from plugins.distribution.reconstruction import algorithm2, to_rooted_order


def test_registers_as_a_generator():
    assert "feedergan-improved" in available("generator")


def test_is_a_joint_feeder_generator():
    gen = FeederGANImproved(num_nodes=24)
    assert isinstance(gen, FeederGenerator)
    feeders = gen.sample(5)
    assert len(feeders) == 5
    for f in feeders:
        assert f.num_nodes == 24
        assert {"length", "ampacity", "phase"} <= set(f.attributes)


def test_algorithm2_gives_each_node_at_most_one_parent():
    soft = torch.softmax(torch.randn(3, 20, 20), dim=1)
    binary = algorithm2(soft)
    assert (binary.sum(1) <= 1).all()                  # in-degree 0 or 1
    assert (torch.diagonal(binary, dim1=1, dim2=2) == 0).all()


def test_rooted_order_puts_root_first_and_parents_before_children():
    # tree rooted at node 3:  3 -> 0, 3 -> 2, 0 -> 1
    a = np.zeros((4, 4), dtype=np.float32)
    a[3, 0] = a[3, 2] = a[0, 1] = 1.0
    relabelled, order = to_rooted_order(a)
    assert order[0] == 3
    assert is_radial_tree(relabelled)
    assert np.all(np.triu(relabelled, 1) == relabelled)   # parent index < child index


def test_non_trees_are_returned_unchanged():
    a = np.zeros((3, 3), dtype=np.float32)
    a[0, 1] = a[1, 2] = a[2, 1] = 1.0                    # node 1 has two parents
    out, order = to_rooted_order(a)
    assert order is None and np.array_equal(out, a)


def test_tree_check_rejects_forests():
    tree = np.zeros((3, 3), dtype=np.float32); tree[0, 1] = tree[0, 2] = 1.0
    forest = np.zeros((3, 3), dtype=np.float32); forest[0, 1] = 1.0
    assert is_radial_tree(tree) and not is_radial_tree(forest)


def test_encoding_round_trips_physical_values():
    m = 6
    feeder = Feeder(np.eye(m, k=1, dtype=np.float32), {
        "length": np.array([0.01, 0.05, 0.1, 0.2, 0.5, 0.79]),
        "ampacity": AMPACITIES[[0, 3, 5, 8, 12, 16]],
        "phase": np.array([6, 6, 0, 1, 2, 3]),
        "distance": np.linspace(0.1, 3.0, m),
        "pseudo_load": np.array([2000, 1500, 800, 400, 100, 25.0]),
        "level": np.arange(m),
    })
    numeric, labels = encode(feeder)
    one_hot = np.concatenate([np.eye(v)[labels[:, k]]
                              for k, v in enumerate(CAT_VOCAB_SIZES)], axis=1)
    out = decode(numeric, one_hot)
    for name in ("length", "distance", "pseudo_load"):
        assert np.allclose(out[name], feeder.attributes[name], rtol=1e-4), name
    for name in ("ampacity", "phase", "level"):
        assert np.array_equal(out[name], feeder.attributes[name]), name


def test_lengths_above_800_m_are_clipped():
    feeder = Feeder(np.eye(2, k=1, dtype=np.float32), {
        "length": np.array([0.1, 2.0]), "ampacity": AMPACITIES[:2],
        "phase": np.array([6, 6]), "distance": np.array([0.1, 2.1]),
        "pseudo_load": np.array([100.0, 50.0]), "level": np.array([0, 0])})
    numeric, _ = encode(feeder)
    assert numeric[1, 0] == pytest.approx(1.0)
