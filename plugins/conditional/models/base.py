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


from abc import ABC, abstractmethod

import numpy as np

from plugins.conditional.encoding import ContextEncoder, OutcomeVocabulary


class ConditionalModel(ABC):
    """P(ampacity, phase | context) for one node."""

    name: str = ""

    # every model has one context encoder, which is shared by training, generation, and evaluation.
    def __init__(self, encoder: ContextEncoder):
        self.encoder = encoder

    # The vocabulary is shared by training, generation, and evaluation., but it is fitted on the training split and stored with the checkpoint.
    @property
    def vocabulary(self) -> OutcomeVocabulary:
        return self.encoder.vocabulary
    
    @abstractmethod
    def fit(self, fields: dict[str, np.ndarray], targets: dict[str, np.ndarray]) -> "ConditionalModel":
        ...

    @abstractmethod
    def log_prob(self, fields, targets) -> np.ndarray: ...

    @abstractmethod
    def sample(self, fields, rng: np.random.Generator) -> np.ndarray: ...

    @abstractmethod
    def state(self) -> dict: ...

    @classmethod
    @abstractmethod
    def from_state(cls, state: dict) -> "ConditionalModel": ...

    # helper
    def bits(self, fields, targets) -> np.ndarray:
        """Return the number of bits required to encode each target given its context."""
        return -self.log_prob(fields, targets) / np.log(2)