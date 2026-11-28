"""
Tests for the conditional ampacity-phase plugin.

The encoding and the back-off table are checked on small hand-built inputs, so
most tests run without the dataset; the ones that need it are skipped when it is
unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridgen import AttributeGenerator, Feeder
from plugins.conditional.config import LADDER, ContextConfig, MLPConfig, TableConfig
from plugins.conditional.encoding import (N_PHASES, ContextEncoder, OutcomeVocabulary,
                                          TreeStructure)
from plugins.conditional.generator import ConditionalAttributes, walk
from plugins.conditional.models import load_model
from plugins.conditional.models.base import ConditionalModel
from plugins.conditional.models.mlp import MLPModel, validation_rows
from plugins.conditional.models.table import (BackoffTable, backoff_probs, count_table,
                                              lookup_counts, marginal)
from plugins.conditional.training import fit_mlp, fit_table
from plugins.distribution.data import schema


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


# ── table: counting and back-off ─────────────────────────────────────────────
def test_count_table_counts_repeated_pairs():
    ids, counts = count_table(np.array([7, 3, 7, 7, 3]), np.array([1, 0, 1, 2, 0]), 3)
    assert ids.tolist() == [3, 7]
    assert counts.tolist() == [[2, 0, 0], [0, 2, 1]]


def test_lookup_gives_zeros_for_unseen_contexts():
    ids, counts = count_table(np.array([7, 3, 7, 7, 3]), np.array([1, 0, 1, 2, 0]), 3)
    rows = lookup_counts(ids, counts, np.array([7, 5, 3, 9]))
    assert rows.tolist() == [[0, 2, 1], [0, 0, 0], [2, 0, 0], [0, 0, 0]]


def test_marginal_is_smoothed_and_never_zero():
    np.testing.assert_allclose(marginal(np.array([0] * 6 + [1] * 3 + [2]), 3),
                               np.array([6.5, 3.5, 1.5]) / 11.5)
    assert np.all(marginal(np.array([0, 0, 2]), 4) > 0)        # outcomes 1 and 3 never occur


def test_backoff_stops_at_the_finest_seen_level():
    outcomes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2])
    level1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    level2 = np.array([0, 0, 0, 0, 0, 10, 10, 10, 11, 11])
    tables = [count_table(level1, outcomes, 3), count_table(level2, outcomes, 3)]
    probs = backoff_probs(marginal(outcomes, 3), tables,
                          [np.array([1, 1, 5]), np.array([11, 99, 99])], beta=5.0)
    np.testing.assert_allclose(probs, [[0.171, 0.568, 0.261],     # seen at both levels
                                       [0.239, 0.596, 0.165],     # seen at level 1 only
                                       [0.478, 0.391, 0.130]],    # seen at neither: marginal
                               atol=1e-3)


# ── table: the model ─────────────────────────────────────────────────────────
def _fitted_table(factorised=False, beta=5.0):
    f = small_tree()
    enc = ContextEncoder(OutcomeVocabulary().fit([f]))
    fields, targets = enc.encode_feeders([f], np.random.default_rng(0))
    model = BackoffTable(enc, TableConfig(beta=beta, factorised=factorised)).fit(fields, targets)
    return model, fields, targets


def test_table_implements_the_model_contract():
    model, _, _ = _fitted_table()
    assert isinstance(model, ConditionalModel)
    assert model.name == "table"
    assert list(model.tables) == ["outcome"]
    assert list(_fitted_table(factorised=True)[0].tables) == ["ampacity_class", "phase"]


def test_unseen_context_backs_off_to_the_marginal():
    model, fields, _ = _fitted_table()
    table = model.tables["outcome"]
    # a parent (160 A, ABC) that no node of the small tree has: every level is unseen
    unseen = dict(fields, parent=np.full_like(fields["parent"],
                                              model.vocabulary.value_code([160.0], [6.0])[0]))
    for f in (fields, unseen):
        np.testing.assert_allclose(table.probs(model.encoder, f).sum(1), 1.0)
    np.testing.assert_allclose(table.probs(model.encoder, unseen),
                               np.tile(table.marginal_probs, (len(unseen["parent"]), 1)))


def test_tiny_beta_reproduces_the_counts():
    model, fields, targets = _fitted_table(beta=1e-9)
    context = model.encoder.index(fields)                     # the finest level
    probs = model.tables["outcome"].probs(model.encoder, fields)
    for row in range(len(context)):
        same = targets["outcome"][context == context[row]]
        freq = np.bincount(same, minlength=model.vocabulary.n_outcomes) / len(same)
        np.testing.assert_allclose(probs[row], freq, atol=1e-6)


def test_sampling_matches_the_probabilities_and_is_reproducible():
    model, fields, _ = _fitted_table()
    one = {name: np.repeat(values[1:2], 20000) for name, values in fields.items()}
    codes = model.sample(one, np.random.default_rng(1))
    assert np.isin(codes, model.vocabulary.pairs).all()       # joint: only real pairs
    freq = np.bincount(model.vocabulary.outcome_of_code(codes),
                       minlength=model.vocabulary.n_outcomes) / len(codes)
    probs = model.tables["outcome"].probs(model.encoder, one)[0]
    assert np.abs(freq - probs).max() < 0.02
    np.testing.assert_array_equal(model.sample(one, np.random.default_rng(7)),
                                  model.sample(one, np.random.default_rng(7)))
    assert model.sample({k: v[:0] for k, v in fields.items()},
                        np.random.default_rng(0)).dtype.kind == "i"     # empty input: int codes


def test_joint_and_factorised_handle_unseen_pairs():
    joint, fields, targets = _fitted_table()
    unseen = dict(targets, outcome=targets["outcome"].copy())
    unseen["outcome"][0] = -1
    with pytest.raises(ValueError):
        joint.log_prob(fields, unseen)

    fact, _, _ = _fitted_table(factorised=True)
    assert np.isfinite(fact.log_prob(fields, unseen)).all()  # scores A and P separately
    many = {name: np.repeat(values, 2000) for name, values in fields.items()}
    codes = fact.sample(many, np.random.default_rng(0))
    assert np.all((codes >= 0) & (codes < fact.vocabulary.n_value_codes))
    assert np.any(fact.vocabulary.outcome_of_code(codes) == -1)   # draws unseen pairs


def test_unfitted_table_raises():
    model, fields, targets = _fitted_table()
    empty = BackoffTable(model.encoder)
    for call in (lambda: empty.log_prob(fields, targets),
                 lambda: empty.sample(fields, np.random.default_rng(0)),
                 empty.state):
        with pytest.raises(RuntimeError):
            call()


@pytest.mark.parametrize("factorised", [False, True])
def test_table_survives_a_checkpoint(tmp_path, factorised):
    """Saved with torch and loaded in its safe mode, the model is unchanged."""
    torch = pytest.importorskip("torch")
    model, fields, targets = _fitted_table(factorised=factorised)
    path = tmp_path / "table.pt"
    torch.save(model.state(), path)
    loaded = BackoffTable.from_state(torch.load(path, weights_only=True))
    assert loaded.config == model.config
    assert loaded.encoder.config == model.encoder.config
    np.testing.assert_array_equal(loaded.bits(fields, targets), model.bits(fields, targets))
    np.testing.assert_array_equal(loaded.sample(fields, np.random.default_rng(0)),
                                  model.sample(fields, np.random.default_rng(0)))


# ── training and loading ─────────────────────────────────────────────────────
@pytest.mark.parametrize("rung", ["t1", "t5"])
def test_fit_table_writes_a_checkpoint_that_load_model_rebuilds(tmp_path, rung):
    """The checkpoint alone is enough: the loaded encoder scores like the fitted model."""
    context, table = LADDER[rung]
    model, path = fit_table([small_tree()], context, table, out_dir=str(tmp_path / "out"),
                            name=rung)
    assert path == str(tmp_path / "out" / f"{rung}.pt")          # out_dir is created
    loaded = load_model(path)
    assert isinstance(loaded, BackoffTable)
    assert loaded.encoder.config == context and loaded.config == table
    fields, targets = loaded.encoder.encode_feeders([small_tree()], np.random.default_rng(0))
    np.testing.assert_array_equal(loaded.bits(fields, targets), model.bits(fields, targets))


def test_fit_table_is_deterministic_for_a_seed(tmp_path):
    a, _ = fit_table([small_tree()], out_dir=str(tmp_path), name="a", seed=3)
    b, _ = fit_table([small_tree()], out_dir=str(tmp_path), name="b", seed=3)
    assert a.state() == b.state()


def test_load_model_rejects_an_unknown_model(tmp_path):
    torch = pytest.importorskip("torch")
    path = tmp_path / "unknown.pt"
    torch.save({"model": "not-a-model"}, path)
    with pytest.raises(ValueError, match="unknown model"):
        load_model(str(path))


# ── MLP ──────────────────────────────────────────────────────────────────────
def _encoded_small_tree(context=None):
    f = small_tree()
    enc = ContextEncoder(OutcomeVocabulary().fit([f]), context)
    fields, targets = enc.encode_feeders([f], np.random.default_rng(0))
    return enc, fields, targets


def test_untrained_mlp_scores_and_samples_without_touching_torch_state():
    """Untrained is a valid, near-uniform model; building it leaves the global torch RNG alone."""
    import torch
    enc, fields, targets = _encoded_small_tree()
    before = torch.random.get_rng_state()
    cuda_before = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    model = MLPModel(enc)
    assert torch.equal(torch.random.get_rng_state(), before)
    if cuda_before is not None:                          # manual_seed seeds CUDA too
        assert torch.equal(torch.cuda.get_rng_state(), cuda_before)
    bits = model.bits(fields, targets)
    assert bits.shape == targets["outcome"].shape
    assert bits.mean() == pytest.approx(np.log2(enc.vocabulary.n_outcomes), abs=0.5)
    np.testing.assert_array_equal(bits, model.bits(fields, targets))     # dropout off
    codes = model.sample(fields, np.random.default_rng(0))
    assert np.isin(codes, enc.vocabulary.pairs).all()


def test_mlp_fit_lowers_the_training_bits():
    enc, fields, targets = _encoded_small_tree()
    untrained = MLPModel(enc).bits(fields, targets).mean()
    model = MLPModel(enc, MLPConfig(max_epochs=200, patience=200, batch_size=4))
    model.fit(fields, targets)
    assert model.bits(fields, targets).mean() < 0.5 * untrained
    assert len(model.history) == 200


def test_mlp_fit_is_deterministic_for_a_seed():
    enc, fields, targets = _encoded_small_tree()
    cfg = MLPConfig(max_epochs=20, batch_size=4, seed=3)
    a = MLPModel(enc, cfg).fit(fields, targets)
    b = MLPModel(enc, cfg).fit(fields, targets)
    np.testing.assert_array_equal(a.bits(fields, targets), b.bits(fields, targets))


def test_mlp_survives_a_checkpoint(tmp_path):
    """Saved with torch and loaded in its safe mode through load_model, the model is unchanged."""
    import torch
    enc, fields, targets = _encoded_small_tree(LADDER["t3"][0])
    model = MLPModel(enc, MLPConfig(max_epochs=5, batch_size=4)).fit(fields, targets)
    path = tmp_path / "mlp.pt"
    torch.save(model.state(), path)
    loaded = load_model(str(path))
    assert isinstance(loaded, MLPModel)
    assert loaded.config == model.config and loaded.encoder.config == model.encoder.config
    assert loaded.history == model.history
    np.testing.assert_array_equal(loaded.bits(fields, targets), model.bits(fields, targets))
    np.testing.assert_array_equal(loaded.sample(fields, np.random.default_rng(0)),
                                  model.sample(fields, np.random.default_rng(0)))


def test_fit_mlp_writes_a_checkpoint_that_load_model_rebuilds(tmp_path):
    context = LADDER["t5"][0]
    model, path = fit_mlp([small_tree()], context, MLPConfig(max_epochs=5, batch_size=4),
                          out_dir=str(tmp_path / "out"), name="mlp_t5")
    assert path == str(tmp_path / "out" / "mlp_t5.pt")
    loaded = load_model(path)
    assert isinstance(loaded, MLPModel) and loaded.encoder.config == context
    fields, targets = loaded.encoder.encode_feeders([small_tree()], np.random.default_rng(0))
    np.testing.assert_array_equal(loaded.bits(fields, targets), model.bits(fields, targets))


def test_validation_rows_hold_back_whole_feeders():
    feeder = np.repeat(np.arange(10), 5)                 # 10 feeders of 5 rows
    held_back = validation_rows(feeder, 0.2, np.random.default_rng(0))
    assert held_back.sum() == 10                         # 2 whole feeders
    assert all(held_back[feeder == f].all() or not held_back[feeder == f].any()
               for f in range(10))
    assert validation_rows(feeder[:5], 0.2, np.random.default_rng(0)) is None   # one feeder


# ── generator ────────────────────────────────────────────────────────────────
class CopyParent:
    """Fake model: the root gets code 5, every other node its parent's code.

    If the walk generates every node after its parent, every node ends up 5.
    """

    def __init__(self, encoder):
        self.root = encoder.root
        self.rows = 0

    def sample(self, fields, rng):
        self.rows += len(fields["parent"])
        parent = fields["parent"]
        return np.where(parent == self.root, 5, parent)


class TorchLengths(AttributeGenerator):
    """Fake length source that samples with torch, like the staged MDN."""

    @property
    def schema(self):
        return schema()

    def generate(self, adjacencies, *, category=None, rng=None):
        import torch
        return [{"length": torch.rand(len(a)).numpy()} for a in adjacencies]


def random_tree(m=128, seed=0):
    """A random rooted tree in BFS order (every parent before its children)."""
    rng = np.random.default_rng(seed)
    parents = [-1] + [int(rng.integers(0, j)) for j in range(1, m)]
    return tree_feeder(parents, [400] * m, [6] * m).adjacency


def test_walk_generates_every_node_after_its_parent():
    f = small_tree()
    enc = ContextEncoder(OutcomeVocabulary().fit([f]))
    m = 128
    trees = [f.adjacency, f.adjacency]
    big = [tree_feeder([-1] + list(range(m - 1)), [400] * m, [6] * m).adjacency,   # chain
           tree_feeder([-1] + [0] * (m - 1), [400] * m, [6] * m).adjacency,        # star
           random_tree(m)]
    for batch in (trees, big):
        model = CopyParent(enc)
        codes = walk(enc, model, batch, np.random.default_rng(0))
        assert np.all(codes == 5)                       # complete and top down
        assert model.rows == codes.size                 # every node exactly once


def test_conditional_attributes_produce_schema_conforming_feeders():
    model, _, _ = _fitted_table()
    gen = ConditionalAttributes(model, lengths=TorchLengths(), batch_size=3)
    assert isinstance(gen, AttributeGenerator)
    trees = [random_tree(seed=s) for s in range(7)]   # 7 trees in batches of 3, 3, 1
    out = gen.generate(trees, rng=np.random.default_rng(0))
    assert len(out) == len(trees)
    for adjacency, attributes in zip(trees, out):
        assert set(attributes) == {"length", "ampacity", "phase"}
        assert all(v.dtype == np.float32 for v in attributes.values())
        assert Feeder(adjacency, attributes).validate(gen.schema) == []
        assert np.isin(attributes["ampacity"], model.vocabulary.ampacity_classes).all()


def test_conditional_attributes_are_reproducible():
    model, _, _ = _fitted_table()
    gen = ConditionalAttributes(model, lengths=TorchLengths())
    trees = [random_tree(seed=s) for s in range(3)]
    a = gen.generate(trees, rng=np.random.default_rng(0))
    b = gen.generate(trees, rng=np.random.default_rng(0))
    c = gen.generate(trees, rng=np.random.default_rng(1))
    for x, y in zip(a, b):                              # same seed: identical, lengths too
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])
    assert any(not np.array_equal(x["length"], z["length"]) for x, z in zip(a, c))


def test_conditional_attributes_reject_a_category():
    model, _, _ = _fitted_table()
    gen = ConditionalAttributes(model, lengths=TorchLengths())
    with pytest.raises(ValueError):
        gen.generate([random_tree()], category="rural")


def test_importing_the_plugin_registers_its_components():
    import plugins.conditional                          # noqa: F401  (registers)
    from gridgen.core.registry import available
    assert "conditional" in available("attribute_generator")
    assert "staged-conditional" in available("generator")


# ── data (skipped without the dataset) ───────────────────────────────────────
def _split(name):
    from plugins.distribution.data import TensorFeederSource
    try:
        return list(TensorFeederSource(split=name))
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"dataset unavailable: {exc}")


def _train():
    return _split("train")


def test_training_vocabulary_matches_the_analysis():
    """17 ampacity classes and 28 observed pairs on the training split."""
    voc = OutcomeVocabulary().fit(_train())
    assert voc.n_ampacity == 17
    assert voc.n_outcomes == 28


#: held-out bits per node on the test split, beta = 5 (analysis/edge_conditions.ipynb and
#: custom-approach results/2026-11-27_wp1_classical_baseline)
EXPECTED_BITS = {"t1": 1.745, "t2": 1.264, "t3": 1.152, "t4": 1.052, "t5": 1.024}


def test_table_ladder_matches_the_analysis():
    """The presets in config.LADDER reproduce the prototype's held-out bits."""
    train, test = _split("train"), _split("test")
    voc = OutcomeVocabulary().fit(train)
    for rung, (context, table) in LADDER.items():
        enc = ContextEncoder(voc, context)
        fields, targets = enc.encode_feeders(train, np.random.default_rng(0))
        test_fields, test_targets = enc.encode_feeders(test, np.random.default_rng(0))
        model = BackoffTable(enc, table).fit(fields, targets)
        bits = model.bits(test_fields, test_targets).mean()
        assert bits == pytest.approx(EXPECTED_BITS[rung], abs=1e-3), rung


def test_t5_table_on_real_topologies_matches_the_prototype():
    """T5 with staged-MDN lengths on the comparison topologies (custom-approach WP1 run).

    Real feeders: 54 % single-phase segments; the prototype's T5: 53.5 %, and
    99.9 % of feeders phase-valid without any constraint.
    """
    import os
    from plugins.distribution import config as shared
    from plugins.distribution.constraints import PhaseSubset
    from plugins.distribution.data import TensorFeederSource
    from plugins.staged.attributes import MDNAttributes
    checkpoint = os.path.join(shared.CHECKPOINT_ROOT, "heldout", "staged", "attributes.pt")
    if not os.path.exists(checkpoint):
        pytest.skip(f"staged checkpoint unavailable: {checkpoint}")
    train = _split("train")
    topologies = [np.asarray(f.adjacency)
                  for f in TensorFeederSource(split="comparison").sample(256, seed=0)]
    voc = OutcomeVocabulary().fit(train)
    enc = ContextEncoder(voc)
    model = BackoffTable(enc).fit(*enc.encode_feeders(train, np.random.default_rng(0)))
    gen = ConditionalAttributes(model, lengths=MDNAttributes(checkpoint=checkpoint))
    feeders = [Feeder(a, x) for a, x in
               zip(topologies, gen.generate(topologies, rng=np.random.default_rng(0)))]
    single = np.mean([np.isin(f.attributes["phase"][1:], [0, 1, 2]).mean() for f in feeders])
    assert single == pytest.approx(0.54, abs=0.03)
    assert PhaseSubset().validate(feeders).feeder_rate >= 0.99
