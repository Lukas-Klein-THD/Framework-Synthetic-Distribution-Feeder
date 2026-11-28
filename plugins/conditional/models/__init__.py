import torch

from plugins.conditional.models.base import ConditionalModel
from plugins.conditional.models.table import BackoffTable


MODELS = {"table": BackoffTable}


def load_model(path: str) -> ConditionalModel:
    state = torch.load(path, weights_only=True)
    return MODELS[state["model"]].from_state(state)