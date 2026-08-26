"""
Tests for the reference plugin.

Anything needing the dataset is skipped when it is unavailable, so the suite
still runs on a clean checkout. The tests that matter most check contracts —
that the plugin's components satisfy the framework interfaces and that generated
feeders conform to the declared schema.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridgen import (
    AttributeGenerator, Constraint, DataSource, Feeder, FeederGenerator,
    StagedGenerator, TopologyGenerator, available)
from plugins import feedergan
from plugins.feedergan import config, schema
from plugins.feedergan.constraints import AmpacityTapering, PhaseSubset
from plugins.feedergan.derived import TopologicalAttributes
from plugins.feedergan.profile import PROFILE_FIELDS, FeederProfile, compute_raw
from plugins.feedergan.topology import TreeGANTopology, harden


def chain_feeder(m: int = 8, phase: int = 6, ampacity: float = 300.0) -> Feeder:
    """A simple valid feeder: one chain, all conductors, uniform attributes."""
    adjacency = np.zeros((m, m), dtype=np.float32)
    for j in range(1, m):
        adjacency[j - 1, j] = 1.0
    return Feeder(adjacency, {
        "device_type": np.zeros(m, dtype=np.float32),
        "length": np.full(m, 0.05, dtype=np.float32),
        "ampacity": np.full(m, ampacity, dtype=np.float32),
        "phase": np.full(m, float(phase), dtype=np.float32),
    })


# ── registration ─────────────────────────────────────────────────────────────
def test_plugin_registers_its_components():
    assert "staged-feedergan" in available("generator")
    assert "tree-gan" in available("topology_generator")
    assert "mdn" in available("attribute_generator")
    assert {"phase-subset", "ampacity-tapering"} <= set(available("constraint"))


# ── topology ─────────────────────────────────────────────────────────────────
def test_generator_produces_valid_radial_trees():
    gen = TreeGANTopology(config.TopologyConfig(num_nodes=16))
    assert isinstance(gen, TopologyGenerator)
    adjacencies = gen.sample_topology(3)
    for a in adjacencies:
        problems = Feeder(a).validate(schema(16).subset([]))
        assert problems == [], problems


def test_generator_is_size_agnostic():
    """No parameter is indexed by node count, so any M works."""
    gen = TreeGANTopology(config.TopologyConfig(num_nodes=12))
    small = gen.sample_topology(2)
    gen.cfg = config.TopologyConfig(num_nodes=40)
    large = gen.sample_topology(2)
    assert small[0].shape == (12, 12) and large[0].shape == (40, 40)


def test_hardening_yields_exactly_one_parent_per_node():
    soft = np.random.default_rng(0).random((2, 6, 6)).astype(np.float32)
    hard = harden(__import__("torch").as_tensor(soft)).numpy()
    for b in range(2):
        assert hard[b].sum() == 5                     # M - 1 edges
        assert (hard[b].sum(axis=0)[1:] == 1).all()   # one parent each
        assert hard[b].sum(axis=0)[0] == 0            # root has none


# ── derived attributes ───────────────────────────────────────────────────────
def test_derived_attributes_are_exact():
    feeder = chain_feeder(5)
    derived = TopologicalAttributes().compute(feeder)
    assert np.allclose(derived["depth"], [0, 1, 2, 3, 4])
    assert np.allclose(derived["distance"], [0.0, 0.05, 0.10, 0.15, 0.20])
    assert np.allclose(derived["n_descendants"], [4, 3, 2, 1, 0])


# ── constraints ──────────────────────────────────────────────────────────────
def test_phase_subset_accepts_valid_and_rejects_invalid():
    assert PhaseSubset().check(chain_feeder(phase=6)).all()      # abc throughout
    bad = chain_feeder(phase=0)                                  # all 'a'
    phase = np.array(bad.attributes["phase"]); phase[2] = 3      # 'ab' under 'a'
    assert not PhaseSubset().check(bad.with_attributes(phase=phase)).all()


def test_phase_repair_reaches_validity():
    bad = chain_feeder(phase=0)
    phase = np.array(bad.attributes["phase"]); phase[2] = 6      # 'abc' under 'a'
    bad = bad.with_attributes(phase=phase)
    repaired = PhaseSubset().repair(bad)
    assert PhaseSubset().check(repaired).all()


def test_constraints_declare_hard_or_soft():
    assert PhaseSubset().hard is True
    assert AmpacityTapering().hard is False


def test_ampacity_tapering_ignores_non_conductor_edges():
    feeder = chain_feeder(4)
    device = np.array(feeder.attributes["device_type"]); device[1:] = 1.0  # switches
    flags = AmpacityTapering().check(feeder.with_attributes(device_type=device))
    assert flags.size == 0            # no conductor->conductor edge remains


def test_validation_separates_element_and_feeder_rates():
    ok, bad = chain_feeder(phase=6), chain_feeder(phase=0)
    phase = np.array(bad.attributes["phase"]); phase[3] = 6
    result = PhaseSubset().validate([ok, bad.with_attributes(phase=phase)])
    assert result.feeder_rate == 0.5          # one of two feeders is clean
    assert 0.0 < result.element_rate < 1.0


# ── profile ──────────────────────────────────────────────────────────────────
def test_profile_summarises_a_feeder():
    raw = compute_raw(chain_feeder(10).attributes)
    assert raw.shape == (len(PROFILE_FIELDS),)
    assert raw[0] == 0.0                       # no switches in a conductor chain


def test_profile_sampling_respects_category():
    feeders = [chain_feeder().with_metadata(category=c)
               for c in ("rural", "rural", "urban-suburban")]
    profile = FeederProfile.fit(feeders)
    assert profile.available_categories == ["rural", "urban-suburban"]
    assert profile.sample(4, category="rural").shape == (4, profile.dim)
    with pytest.raises(ValueError, match="no profiles for category"):
        profile.sample(1, category="industrial")


# ── composition ──────────────────────────────────────────────────────────────
def test_staged_pipeline_is_a_feeder_generator():
    gen = StagedGenerator(TreeGANTopology(config.TopologyConfig(num_nodes=16)),
                          attributes=None, derived=[])
    assert isinstance(gen, FeederGenerator)
    assert gen.sample(2)[0].num_nodes == 16


# ── data (skipped without the dataset) ───────────────────────────────────────
def _source():
    from plugins.feedergan.data import SourcePathLabeler, TensorFeederSource
    try:
        return TensorFeederSource(labeler=SourcePathLabeler(),
                                  include_decomposed=False, max_graphs=8)
    except (FileNotFoundError, RuntimeError) as exc:
        pytest.skip(f"dataset unavailable: {exc}")


def test_data_source_yields_schema_conforming_feeders():
    source = _source()
    assert isinstance(source, DataSource)
    feeder = source[0]
    assert feeder.validate(source.schema) == []
    assert feeder.num_nodes == config.NUM_NODES


def test_real_feeders_satisfy_the_hard_constraint():
    """Calibration check: phase subset must hold exactly on real data."""
    source = _source()
    result = PhaseSubset().calibrate(list(source))
    assert result.element_rate == pytest.approx(1.0)
    assert result.feeder_rate == pytest.approx(1.0)
