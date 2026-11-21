"""
Domain-specific evaluation for distribution feeders: the attribute encoder and
the scorecard.

The generic graph metrics (descriptors, distances and the two suites) live in
:mod:`plugins.graph_metrics`; this module adds what only makes sense for
feeders:

* :class:`AttributeEncoder` turns line length, ampacity and phase into the
  per-node vectors of the attributed embedding.
* ``realism_score``, ``physical_score`` and ``operational_score`` are the three
  dimensions of the scorecard, each on a 0-10 scale.

The generic names are re-exported below, so imports written against the
earlier layout (``from plugins.distribution.metrics import RandomGINEmbedding``)
keep working.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gridgen.core.feeder import Feeder

# moved to plugins.graph_metrics; re-exported for compatibility
from plugins.graph_metrics import (  # noqa: F401
    DegreeHistogram, FrechetDistance, LaplacianSpectrum, MaximumMeanDiscrepancy,
    PrecisionRecallDensityCoverage, RandomGINEmbedding, attributed_metrics,
    is_radial_tree, structural_metrics)


# ── node attributes as GIN input ─────────────────────────────────────────────
class AttributeEncoder:
    """Per-node numeric vector for the attributed embedding.

    ``length`` is log10-scaled and standardised, ``ampacity`` one-hot over the
    observed ratings (nearest match), ``phase`` one-hot over the seven codes.
    Fit it on **real** feeders and use the same instance for every generated
    set, so all models are embedded identically.
    """

    def __init__(self, ampacities, log_mean: float, log_std: float,
                 n_phase: int = 7, eps: float = 1e-3):
        self.ampacities = np.asarray(ampacities, dtype=float)
        self.log_mean, self.log_std = float(log_mean), float(log_std)
        self.n_phase, self.eps = n_phase, eps

    @property
    def dim(self) -> int:
        return 1 + len(self.ampacities) + self.n_phase

    @classmethod
    def fit(cls, feeders: Sequence[Feeder], eps: float = 1e-3) -> "AttributeEncoder":
        length = np.concatenate([np.asarray(f.attributes["length"], float) for f in feeders])
        amps = np.concatenate([np.asarray(f.attributes["ampacity"], float) for f in feeders])
        log_length = np.log10(np.clip(length, 0.0, None) + eps)
        return cls(np.unique(np.round(amps, 3)), log_length.mean(),
                   log_length.std() + 1e-8, eps=eps)

    def __call__(self, feeder: Feeder) -> np.ndarray:
        length = np.asarray(feeder.attributes["length"], float)
        amps = np.asarray(feeder.attributes["ampacity"], float)
        phase = np.clip(np.asarray(feeder.attributes["phase"]).astype(int), 0, self.n_phase - 1)
        log_length = (np.log10(np.clip(length, 0.0, None) + self.eps)
                      - self.log_mean) / self.log_std
        amp_idx = np.abs(amps[:, None] - self.ampacities[None, :]).argmin(1)
        return np.concatenate([log_length[:, None],
                               np.eye(len(self.ampacities))[amp_idx],
                               np.eye(self.n_phase)[phase]], axis=1)


# ── scorecard: three dimensions on a 0-10 scale ──────────────────────────────
def realism_score(precision: float, recall: float,
                  ref_precision: float, ref_recall: float) -> float:
    """Realism on a 0-10 scale: F1 of precision and recall, relative to real vs real.

    The harmonic mean drops to zero when either term does, so high precision
    cannot hide a collapse in recall.  Dividing by the real-vs-real F1 measured
    with the same embedding and sample size anchors the scale: 10 is what an
    independent real sample reaches, 0 is an untrained model.  FD is left out
    on purpose, because its scale depends on the embedding initialisation.
    """
    def f1(p, r):
        return 0.0 if p + r <= 0 else 2.0 * p * r / (p + r)
    ref = f1(ref_precision, ref_recall)
    return 0.0 if ref <= 0 else 10.0 * f1(precision, recall) / ref


def physical_score(valid_edges: float, valid_feeders: float) -> float:
    """Physical consistency on a 0-10 scale against the absolute target of 100 %.

    The mean of the phase-valid edge share and the phase-valid feeder share
    (both fractions).  Real feeders satisfy the phase rule everywhere, so the
    target and the reference coincide.
    """
    return 10.0 * (valid_edges + valid_feeders) / 2.0


def operational_score(energised: float, in_band: float, within_rating: float,
                      unbalance_ok: float) -> float:
    """Operational performance on a 0-10 scale against absolute targets of 100 %.

    ``10 * E * mean(V, W, U)``, all fractions: energised phases E, energised
    phases with a voltage inside the evaluation band V (ANSI C84.1 Range A),
    lines within their rating W, and three-phase buses within the 3 % voltage
    unbalance limit U.  V, W and U are measured on the energised part only, so
    they enter through E: a mostly unfed network cannot score well on the
    voltages and currents of its few energised elements.
    """
    return 10.0 * energised * (in_band + within_rating + unbalance_ok) / 3.0
