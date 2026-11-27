"""
Tests for the conditional ampacity-phase plugin.

The encoding is checked on small hand-built trees, so most tests run without the
dataset; the ones that need it are skipped when it is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridgen import Feeder
from plugins.conditional.config import ContextConfig
from plugins.conditional.encoding import (N_PHASES, ContextEncoder, OutcomeVocabulary,
                                          TreeStructure)


def tree_feeder(parents, ampacity, phase) -> Feeder:
    """A feeder from a parent list (BFS order, root first with parent -1)."""
    m = len(parents)
    adjacency = np.zeros((m, m), dtype=np.float32)
    for j, p in enumerate(parents[1:], start=1):
        adjacency[p, j] = 1.0
    return Feeder(adjacency, {"length": np.full(m, 0.05, dtype=np.float32),
                              "ampacity": np.asarray(ampacity, dtype=np.float32),
                              "phase": np.asarray(phase, dtype=np.float32)})


# root 0 with children 1 (subtree 1), 2 (subtree 5), 3 (subtree 2)
#   2 -> 4 (subtree 3), 5;  4 -> 7, 8 (a tie);  3 -> 6 (an only child)
PARENTS = [-1, 0, 0, 0, 2, 2, 3, 4, 4]
AMPS = [400, 160, 400, 325, 400, 160, 325, 160, 160]
PHASES = [6, 0, 6, 6, 6, 1, 6, 0, 2]


def small_tree():
    return tree_feeder(PARENTS, AMPS, PHASES)


# ── vocabulary ───────────────────────────────────────────────────────────────
def test_vocabulary_round_trip():
    f = small_tree()
    voc = OutcomeVocabulary().fit([f])
    assert voc.ampacity_classes.tolist() == [160.0, 325.0, 400.0]
    assert voc.n_outcomes == len(set(zip(AMPS, PHASES)))
    ampacity, phase = voc.decode(voc.encode(f.attributes["ampacity"], f.attributes["phase"]))
    assert ampacity.dtype == np.float32 and phase.dtype == np.float32
    np.testing.assert_array_equal(ampacity, f.attributes["ampacity"])
    np.testing.assert_array_equal(phase, f.attributes["phase"])


def test_unseen_pair_has_a_value_but_no_outcome():
    voc = OutcomeVocabulary().fit([small_tree()])
    assert voc.encode([160.0], [6.0])[0] == -1                 # 160 A three-phase: unseen
    code = voc.value_code([160.0], [6.0])[0]
    assert 0 <= code < voc.n_value_codes
    assert voc.decode_code([code])[0][0] == 160.0


def test_vocabulary_rejects_unknown_values():
    voc = OutcomeVocabulary().fit([small_tree()])
    with pytest.raises(ValueError):
        voc.ampacity_class([250.0])                           # not a training class
    with pytest.raises(ValueError):
        voc.phase_code([7.0])
    with pytest.raises(ValueError):
        voc.decode([-1])
    with pytest.raises(RuntimeError):
        OutcomeVocabulary().n_outcomes


def test_vocabulary_state_round_trip():
    voc = OutcomeVocabulary().fit([small_tree()])
    copy = OutcomeVocabulary.from_state(voc.state())
    np.testing.assert_array_equal(copy.pairs, voc.pairs)
    np.testing.assert_array_equal(copy.ampacity_classes, voc.ampacity_classes)


# ── tree structure ───────────────────────────────────────────────────────────
def test_siblings_are_ordered_largest_subtree_first():
    f = small_tree()
    tree = TreeStructure([f.adjacency], ContextConfig(), np.random.default_rng(0))
    assert tree.subtree_size[0, :4].tolist() == [9, 1, 5, 2]
    # children of the root: 2 (size 5), 3 (size 2), 1 (size 1)
    assert tree.rank[0, [2, 3, 1]].tolist() == [0, 1, 2]
    assert tree.previous[0, [2, 3, 1]].tolist() == [-1, 2, 3]
    assert tree.other[0, [2, 3, 1]].tolist() == [3, 2, 3]   # first child looks at the next
    assert tree.other[0, 6] == -1 and tree.previous[0, 6] == -1    # only child


def test_ties_are_broken_by_the_rng():
    f = small_tree()                                        # 7 and 8: both leaves under 4
    orders = {tuple(TreeStructure([f.adjacency], ContextConfig(),
                                  np.random.default_rng(s)).rank[0, [7, 8]].tolist())
              for s in range(20)}
    assert orders == {(0, 1), (1, 0)}


def test_chain_has_no_siblings():
    m = 128
    f = tree_feeder([-1] + list(range(m - 1)), [400] * m, [6] * m)
    tree = TreeStructure([f.adjacency], ContextConfig(), np.random.default_rng(0))
    assert np.all(tree.other == -1) and np.all(tree.previous == -1)
    assert tree.subtree_bin[0, 0] == ContextConfig().n_subtree_bins - 1    # 128 >= 33
    assert tree.subtree_bin[0, -1] == 0                                    # leaf


def test_non_bfs_order_is_rejected():
    adjacency = np.zeros((3, 3), dtype=np.float32)
    adjacency[0, 2] = adjacency[2, 1] = 1.0                 # node 1's parent is node 2
    with pytest.raises(ValueError):
        TreeStructure([adjacency], ContextConfig(), np.random.default_rng(0))


# ── context ──────────────────────────────────────────────────────────────────
def test_context_fields_of_a_real_tree():
    f = small_tree()
    voc = OutcomeVocabulary().fit([f])
    enc = ContextEncoder(voc)
    fields, targets = enc.encode_feeders([f], np.random.default_rng(0))
    assert enc.active == ("parent", "subtree", "sibling_subtree", "previous_sibling")
    assert fields["parent"][0] == enc.root
    assert fields["parent"][1] == voc.value_code([400.0], [6.0])[0]
    assert fields["sibling_subtree"][6] == enc.only_child
    assert fields["previous_sibling"][2] == enc.first_child  # largest child of the root
    assert fields["previous_sibling"][3] == targets["outcome"][2]
    assert np.all(targets["outcome"] >= 0)


def test_index_is_unique_and_checked():
    voc = OutcomeVocabulary().fit([small_tree()])
    enc = ContextEncoder(voc)
    fields, _ = enc.encode_feeders([small_tree()], np.random.default_rng(0))
    rows = np.stack([fields[n] for n in enc.active], 1)
    idx = enc.index(fields)
    assert len(np.unique(idx)) == len(np.unique(rows, axis=0))
    assert [len(l) for l in enc.levels()] == [1, 2, 3, 4]
    bad = dict(fields, subtree=fields["subtree"] + enc.radix["subtree"])
    with pytest.raises(ValueError):
        enc.index(bad)


def test_fields_require_top_down_values():
    f = small_tree()
    voc = OutcomeVocabulary().fit([f])
    enc = ContextEncoder(voc)
    tree = enc.structure([f.adjacency], np.random.default_rng(0))
    empty = np.full((1, tree.m), -1)
    with pytest.raises(ValueError):
        enc.fields(tree, empty, empty, np.array([0]), np.array([4]))   # parent 2 not set


def test_config_switches_fields():
    voc = OutcomeVocabulary().fit([small_tree()])
    enc = ContextEncoder(voc, ContextConfig(use_subtree=False, use_sibling_subtree=False,
                                            use_previous_sibling=False))
    assert enc.active == ("parent",)
    assert enc.radix["parent"] == voc.n_ampacity * N_PHASES + 1


# ── data (skipped without the dataset) ───────────────────────────────────────
def _train():
    from plugins.distribution.data import TensorFeederSource
    try:
        return list(TensorFeederSource(split="train"))
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"dataset unavailable: {exc}")


def test_training_vocabulary_matches_the_analysis():
    """17 ampacity classes and 28 observed pairs on the training split."""
    voc = OutcomeVocabulary().fit(_train())
    assert voc.n_ampacity == 17
    assert voc.n_outcomes == 28
