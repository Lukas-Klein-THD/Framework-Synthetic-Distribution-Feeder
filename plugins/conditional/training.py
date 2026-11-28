"""
Fitting the conditional models on training data and saving them as checkpoints.

Kept out of the model modules so that the models stay readable as models, and
so that every model reuses the same data plumbing: the vocabulary is fitted on
the training feeders, the feeders are encoded with a fixed seed (the sibling
tie-breaks), and the fitted model is saved with :meth:`ConditionalModel.state`,
so :func:`plugins.conditional.models.load_model` rebuilds it, encoder included.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from plugins.conditional import config
from plugins.conditional.config import ContextConfig, TableConfig
from plugins.conditional.encoding import ContextEncoder, OutcomeVocabulary
from plugins.conditional.models.table import BackoffTable


def fit_table(source, context: ContextConfig | None = None, table: TableConfig | None = None,
              out_dir: str = config.CHECKPOINT_DIR, name: str = "table",
              seed: int = 0) -> tuple[BackoffTable, str]:
    """Fit a back-off table on ``source`` and save it as ``<out_dir>/<name>.pt``.

    ``source`` is any iterable of feeders, normally ``TensorFeederSource(split="train")``.
    Returns the fitted model and the checkpoint path.
    """
    feeders = list(source)
    vocabulary = OutcomeVocabulary().fit(feeders)
    encoder = ContextEncoder(vocabulary, context)
    fields, targets = encoder.encode_feeders(feeders, np.random.default_rng(seed))
    model = BackoffTable(encoder, table).fit(fields, targets)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pt")
    torch.save(model.state(), path)
    return model, path
