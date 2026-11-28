"""
The conditional models, and loading any of them from a checkpoint.

Every checkpoint names its class in ``state["model"]``, so whoever loads one
does not need to know which model is inside.
"""

from __future__ import annotations

import torch

from plugins.conditional.models.base import ConditionalModel
from plugins.conditional.models.table import BackoffTable

#: ``ConditionalModel.name`` -> class; a new model adds itself here
MODELS: dict[str, type[ConditionalModel]] = {BackoffTable.name: BackoffTable}

__all__ = ["ConditionalModel", "BackoffTable", "MODELS", "load_model"]


def load_model(path: str) -> ConditionalModel:
    """Rebuild a fitted model from a checkpoint; ``state["model"]`` says which class."""
    state = torch.load(path, weights_only=True)
    if state.get("model") not in MODELS:
        raise ValueError(f"unknown model {state.get('model')!r} in {path}; "
                         f"known: {sorted(MODELS)}")
    return MODELS[state["model"]].from_state(state)
