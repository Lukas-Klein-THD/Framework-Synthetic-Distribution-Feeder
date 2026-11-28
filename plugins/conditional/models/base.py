"""
The contract every conditional model implements (table, MLP, QCBM).

A model estimates P(ampacity, phase | context) for one node.  It never sees a
tree: the generator builds the context fields top down and asks the model for
one draw per row, so models are interchangeable without touching the sampler,
the training entry point or the evaluation.

Three choices keep one interface for joint and factorised models alike:

    targets   a dict, as returned by ContextEncoder.encode_feeders.  A joint model
              reads ``outcome``, a factorised one ``ampacity_class`` and ``phase``.
    sample    returns value codes (ampacity_class * N_PHASES + phase), not outcome
              indices: a factorised model can draw a pair never seen in training,
              which has a value code but no outcome.
    sampling  the generator only calls ``sample``; ``log_prob`` is for evaluation.
              A QCBM on hardware yields samples, not probabilities.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from plugins.conditional.encoding import ContextEncoder, OutcomeVocabulary


class ConditionalModel(ABC):
    """P(ampacity, phase | context) for one node.

    The model owns its context encoder (and through it the vocabulary), so
    training, generation and evaluation always encode the same way, and a
    checkpoint carries both.
    """

    #: key in ``state()["model"]``, used by :func:`plugins.conditional.models.load_model`
    name: str = ""

    def __init__(self, encoder: ContextEncoder):
        self.encoder = encoder

    @property
    def vocabulary(self) -> OutcomeVocabulary:
        """The (ampacity, phase) vocabulary, fitted on the training split."""
        return self.encoder.vocabulary

    @abstractmethod
    def fit(self, fields: dict[str, np.ndarray],
            targets: dict[str, np.ndarray]) -> "ConditionalModel":
        """Fit on encoded training rows; returns ``self``."""

    @abstractmethod
    def log_prob(self, fields, targets) -> np.ndarray:
        """Natural-log probability of each row's target, [n_rows]."""

    @abstractmethod
    def sample(self, fields, rng: np.random.Generator) -> np.ndarray:
        """One draw per row, as value codes [n_rows]."""

    @abstractmethod
    def state(self) -> dict:
        """Everything needed to rebuild the model, in plain Python types."""

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> "ConditionalModel":
        """Rebuild a fitted model from :meth:`state`."""

    def bits(self, fields, targets) -> np.ndarray:
        """Bits needed to encode each row's target given its context, [n_rows]."""
        return -self.log_prob(fields, targets) / np.log(2)