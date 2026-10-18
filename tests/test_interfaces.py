"""
Interface-level tests.

These exercise the contracts rather than any model: that a staged pipeline and a
monolithic generator are interchangeable, that schema validation catches
malformed feeders, that constraints aggregate element- and feeder-wise rates
correctly, and that the registry behaves.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridgen import (
    AttributeKind, AttributeSpec, Constraint, DataSource, Descriptor,
    DescriptorMetric, Direction, Distance, DerivedAttributes, Evaluator, Feeder,
    FeederGenerator, FeederSchema, GraphSpec, Pipeline, StagedGenerator,
    TopologyGenerator, AttributeGenerator, available, build, register)

M = 8
SCHEMA = FeederSchema(GraphSpec(num_nodes=M),
                      (AttributeSpec("v", AttributeKind.CONTINUOUS),))


def random_tree(rng, m=M):
    a = np.zeros((m, m))
    for j in range(1, m):
        a[rng.integers(0, j), j] = 1
    return a


# ── fixtures ─────────────────────────────────────────────────────────────────
class DummySource(DataSource):
    def __init__(self, n=24, seed=0):
        rng = np.random.default_rng(seed)
        self._f = [Feeder(random_tree(rng), {"v": rng.random(M)},
                          {"category": "rural" if i % 2 else "urban"})
                   for i in range(n)]

    @property
    def schema(self): return SCHEMA
    def __len__(self): return len(self._f)
    def __getitem__(self, i): return self._f[i]


class DummyTopology(TopologyGenerator):
    categories = ("rural", "urban")

    @property
    def graph_spec(self): return GraphSpec(num_nodes=M)

    def sample_topology(self, n, *, category=None, rng=None):
        rng = rng or np.random.default_rng()
        return [random_tree(rng) for _ in range(n)]


class DummyAttributes(AttributeGenerator):
    categories = ("rural",)          # narrower than the topology stage on purpose

    @property
    def schema(self): return SCHEMA

    def generate(self, adjacencies, *, category=None, rng=None):
        rng = rng or np.random.default_rng()
        return [{"v": rng.random(a.shape[0])} for a in adjacencies]


class Depth(DerivedAttributes):
    @property
    def produces(self): return ("depth",)

    def compute(self, feeder):
        p, d = feeder.parents(), np.zeros(feeder.num_nodes)
        for j in range(1, feeder.num_nodes):
            d[j] = d[p[j]] + 1
        return {"depth": d}


class Monolithic(FeederGenerator):
    @property
    def schema(self): return SCHEMA

    def sample(self, n, *, category=None, rng=None):
        rng = rng or np.random.default_rng()
        return [Feeder(random_tree(rng), {"v": rng.random(M)}) for _ in range(n)]


# ── feeder + schema ──────────────────────────────────────────────────────────
def test_valid_feeder_has_no_problems():
    f = Feeder(random_tree(np.random.default_rng(0)), {"v": np.zeros(M)})
    assert f.validate(SCHEMA) == []


def test_schema_detects_missing_attribute_and_wrong_size():
    f = Feeder(random_tree(np.random.default_rng(0)))
    assert any("missing attribute" in p for p in f.validate(SCHEMA))
    small = Feeder(np.zeros((3, 3)), {"v": np.zeros(3)})
    assert any("expected 8 nodes" in p for p in small.validate(SCHEMA))


def test_radiality_violation_is_reported_not_raised():
    a = random_tree(np.random.default_rng(0))
    a[2, 1] = 1                                     # extra edge -> a cycle
    problems = Feeder(a, {"v": np.zeros(M)}).validate(SCHEMA)
    assert any("edges" in p or "parent" in p for p in problems)


def test_meshed_graph_is_fine_when_declared():
    a = random_tree(np.random.default_rng(0))
    a[2, 1] = 1
    meshed = FeederSchema(GraphSpec(radial=False, num_nodes=M), SCHEMA.attributes)
    assert Feeder(a, {"v": np.zeros(M)}).validate(meshed) == []


# ── generation ───────────────────────────────────────────────────────────────
def test_staged_and_monolithic_satisfy_the_same_interface():
    staged = StagedGenerator(DummyTopology(), DummyAttributes(), [Depth()])
    for gen in (staged, Monolithic()):
        feeders = gen.sample(4)
        assert len(feeders) == 4
        assert all(isinstance(f, Feeder) for f in feeders)
        assert all("v" in f.attributes for f in feeders)


def test_derived_attributes_are_applied():
    staged = StagedGenerator(DummyTopology(), DummyAttributes(), [Depth()])
    assert "depth" in staged.sample(1)[0].attributes


def test_topology_only_generator_is_allowed():
    staged = StagedGenerator(DummyTopology())
    f = staged.sample(1)[0]
    assert f.attributes == {}
    assert f.num_nodes == M


def test_staged_categories_are_the_intersection_of_stages():
    staged = StagedGenerator(DummyTopology(), DummyAttributes())
    assert staged.categories == ("rural",)
    with pytest.raises(ValueError, match="cannot condition"):
        staged.sample(1, category="urban")


def test_category_is_recorded_in_metadata():
    staged = StagedGenerator(DummyTopology(), DummyAttributes())
    assert staged.sample(1, category="rural")[0].category == "rural"


# ── data source ──────────────────────────────────────────────────────────────
def test_source_grouping_and_filtering():
    src = DummySource(n=24)
    assert {k: len(v) for k, v in src.by_category().items()} == {"urban": 12, "rural": 12}
    assert len(src.with_category("rural")) == 12
    assert all(f.category == "rural" for f in src.with_category("rural"))


# ── constraints ──────────────────────────────────────────────────────────────
class AlwaysHalf(Constraint):
    """Satisfied on exactly the even-indexed edges — a known ground truth."""

    @property
    def name(self): return "half"

    def check(self, feeder):
        return np.arange(feeder.num_nodes - 1) % 2 == 0


def test_constraint_reports_element_and_feeder_rates():
    feeders = DummySource(n=5)._f
    result = AlwaysHalf().validate(feeders)
    assert result.element_rate == pytest.approx(4 / 7)
    assert result.feeder_rate == 0.0            # no feeder is fully satisfied
    assert result.n_elements == 5 * (M - 1)


def test_repair_is_optional_and_advertised():
    assert AlwaysHalf().repairable is False
    with pytest.raises(NotImplementedError):
        AlwaysHalf().repair(DummySource(n=1)._f[0])


# ── evaluation ───────────────────────────────────────────────────────────────
class MeanV(Descriptor):
    @property
    def name(self): return "meanv"
    def __call__(self, feeders):
        return np.array([[float(np.mean(f.attributes["v"]))] for f in feeders])


class AbsGap(Distance):
    @property
    def name(self): return "absgap"
    @property
    def direction(self): return Direction.LOWER_IS_BETTER
    def __call__(self, real, generated):
        return {"value": float(abs(real.mean() - generated.mean()))}


def test_scorecard_includes_a_real_vs_real_reference():
    src = DummySource(n=20)
    card = Evaluator([DescriptorMetric(MeanV(), AbsGap())]).run(
        list(src), Monolithic().sample(20), with_reference=True)
    key = "meanv.absgap.value"
    assert key in card.scores and key in card.reference
    assert card.directions[key] is Direction.LOWER_IS_BETTER


def test_pipeline_runs_end_to_end():
    pipe = Pipeline(DummySource(), Monolithic(), [AlwaysHalf()],
                    [DescriptorMetric(MeanV(), AbsGap())])
    report = pipe.run(n=8)
    assert report.scorecard is not None
    assert len(report.validation) == 1 and len(report.calibration) == 1
    assert "half" in str(report)


# ── registry ─────────────────────────────────────────────────────────────────
def test_registry_roundtrip_and_errors():
    register("generator", "dummy-mono")(Monolithic)
    assert "dummy-mono" in available("generator")
    assert isinstance(build("generator", "dummy-mono"), Monolithic)
    with pytest.raises(ValueError, match="already registered"):
        register("generator", "dummy-mono")(Monolithic)
    with pytest.raises(KeyError, match="no generator registered"):
        build("generator", "nope")


def test_realism_score_is_anchored_and_punishes_collapse():
    from plugins.distribution.metrics import realism_score
    assert realism_score(0.98, 0.97, 0.98, 0.97) == pytest.approx(10.0)   # real reference
    assert realism_score(0.0, 0.0, 0.98, 0.97) == 0.0                     # untrained
    assert realism_score(0.98, 0.01, 0.97, 0.95) < 0.3                    # precise but collapsed
    assert realism_score(0.47, 0.94, 0.97, 0.95) == pytest.approx(6.53, abs=0.01)


def test_physical_and_operational_scores_use_absolute_targets_and_gate():
    from plugins.distribution.metrics import operational_score, physical_score
    assert physical_score(1.0, 1.0) == pytest.approx(10.0)
    assert physical_score(0.653, 0.0) == pytest.approx(3.265)          # mean, not product
    assert operational_score(1.0, 1.0, 1.0, 1.0) == pytest.approx(10.0)
    # perfect-looking values on a mostly unfed network must not score well
    assert operational_score(0.096, 1.0, 0.999, 1.0) < 1.0
    assert operational_score(1.0, 0.939, 0.976, 0.977) == pytest.approx(9.64, abs=0.01)
