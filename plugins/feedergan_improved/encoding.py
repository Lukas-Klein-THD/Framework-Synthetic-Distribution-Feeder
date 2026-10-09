"""
Attribute representation of the improved FeederGAN.

The paper scales every numeric attribute linearly to [-1, 1].  That presses most
of the real mass against -1 (a median segment of 66 m in an 800 m range), where
a tanh output has almost no gradient, and it treats a discrete rating as
continuous.  Here each attribute gets the type it has in the data:

    numeric channels      length    log1p(metres), scaled to [-1, 1] over 0-800 m
                          distance  linear, [-1, 1] over the data maximum
                          load      log1p(kVA), scaled to [-1, 1]
    categorical channels  level     15 classes (FeederGAN level)
                          phase     7 codes a..abc
                          ampacity  17 conductor ratings of the corpus

Segments longer than 800 m (the paper's nominal maximum, about 0.6 % of the
data) are clipped to 800 m, as in the thesis model.
"""

from __future__ import annotations

import numpy as np

from gridgen.core.feeder import Feeder

#: The paper's nominal maximum segment length (km).
LENGTH_MAX_KM = 0.8
#: Data maxima over the preprocessed corpus (km, kVA).
DISTANCE_MAX_KM = 42.14043045043945
LOAD_MAX_KVA = 39925.0
#: Every conductor rating present in the corpus (A).
AMPACITIES = np.array([151, 165, 170, 205, 295, 325, 350, 370, 390, 445, 465, 475,
                       525, 550, 575, 595, 960], dtype=float)
N_LEVELS, N_PHASES = 15, 7

NUM_DIM = 3                                       # length, distance, load
CAT_VOCAB_SIZES = (N_LEVELS, N_PHASES, len(AMPACITIES))

_LOG_LENGTH_MAX = np.log1p(LENGTH_MAX_KM * 1000.0)
_LOG_LOAD_MAX = np.log1p(LOAD_MAX_KVA)


def encode(feeder: Feeder) -> tuple[np.ndarray, np.ndarray]:
    """Real feeder -> (numeric [M, 3] in [-1, 1], class labels [M, 3]).

    Needs the stored extras ``distance``, ``pseudo_load`` and ``level``; load
    the data with ``TensorFeederSource(extras=("distance", "pseudo_load", "level"))``.
    """
    a = feeder.attributes
    length_m = np.clip(np.asarray(a["length"], float), 0.0, LENGTH_MAX_KM) * 1000.0
    numeric = np.stack([
        np.log1p(length_m) / _LOG_LENGTH_MAX * 2.0 - 1.0,
        np.asarray(a["distance"], float) / DISTANCE_MAX_KM * 2.0 - 1.0,
        np.log1p(np.clip(np.asarray(a["pseudo_load"], float), 0.0, None))
        / _LOG_LOAD_MAX * 2.0 - 1.0,
    ], axis=1)
    amps = np.round(np.asarray(a["ampacity"], float))
    labels = np.stack([
        np.clip(np.asarray(a["level"]).astype(int), 0, N_LEVELS - 1),
        np.clip(np.asarray(a["phase"]).astype(int), 0, N_PHASES - 1),
        np.abs(amps[:, None] - AMPACITIES[None, :]).argmin(1),
    ], axis=1)
    return numeric.astype(np.float32), labels.astype(np.int64)


def decode(numeric: np.ndarray, categorical: np.ndarray) -> dict[str, np.ndarray]:
    """Generator output of one graph -> named physical attributes.

    numeric      [M, 3] in [-1, 1]
    categorical  [M, sum(CAT_VOCAB_SIZES)], one-hot or probabilities
    """
    numeric = np.asarray(numeric, float)
    parts = np.split(np.asarray(categorical), np.cumsum(CAT_VOCAB_SIZES)[:-1], axis=-1)
    level, phase, amp_idx = (p.argmax(-1) for p in parts)
    return {
        "length": (np.expm1((numeric[:, 0] + 1.0) / 2.0 * _LOG_LENGTH_MAX) / 1000.0
                   ).astype(np.float32),
        "ampacity": AMPACITIES[amp_idx].astype(np.float32),
        "phase": phase.astype(np.float32),
        # generated as well, but not part of the shared schema
        "distance": ((numeric[:, 1] + 1.0) / 2.0 * DISTANCE_MAX_KM).astype(np.float32),
        "pseudo_load": np.expm1((numeric[:, 2] + 1.0) / 2.0 * _LOG_LOAD_MAX
                                ).astype(np.float32),
        "level": level.astype(np.float32),
    }
