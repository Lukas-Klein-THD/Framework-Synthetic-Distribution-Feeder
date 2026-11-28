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
from typing import Callable

import numpy as np
import torch

from plugins.conditional import config
from plugins.conditional.config import ContextConfig, MLPConfig, TableConfig
from plugins.conditional.encoding import ContextEncoder, OutcomeVocabulary
from plugins.conditional.models.base import ConditionalModel
from plugins.conditional.models.mlp import MLPModel
from plugins.conditional.models.table import BackoffTable


def _fit_and_save(source, context: ContextConfig | None,
                  make_model: Callable[[ContextEncoder], ConditionalModel],
                  out_dir: str, name: str, seed: int) -> tuple[ConditionalModel, str]:
    """The plumbing every model shares: vocabulary, encoding, fit, checkpoint.

    ``seed`` drives the encoding (the random order of equal-sized siblings);
    ``make_model`` builds the unfitted model on the encoder.
    """
    feeders = list(source)
    vocabulary = OutcomeVocabulary().fit(feeders)
    encoder = ContextEncoder(vocabulary, context)
    fields, targets = encoder.encode_feeders(feeders, np.random.default_rng(seed))
    model = make_model(encoder).fit(fields, targets)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pt")
    torch.save(model.state(), path)
    return model, path


def fit_table(source, context: ContextConfig | None = None, table: TableConfig | None = None,
              out_dir: str = config.CHECKPOINT_DIR, name: str = "table",
              seed: int = 0) -> tuple[BackoffTable, str]:
    """Fit a back-off table on ``source`` and save it as ``<out_dir>/<name>.pt``.

    ``source`` is any iterable of feeders, normally ``TensorFeederSource(split="train")``.
    Returns the fitted model and the checkpoint path.
    """
    return _fit_and_save(source, context, lambda encoder: BackoffTable(encoder, table),
                         out_dir, name, seed)


def fit_mlp(source, context: ContextConfig | None = None, mlp: MLPConfig | None = None,
            out_dir: str = config.CHECKPOINT_DIR, name: str = "mlp",
            seed: int = 0) -> tuple[MLPModel, str]:
    """Train the network on ``source`` and save it as ``<out_dir>/<name>.pt``.

    ``seed`` drives the encoding as for the table; the network's own randomness
    (initial weights, validation feeders, shuffling, dropout) is ``mlp.seed``.
    Returns the trained model and the checkpoint path.
    """
    return _fit_and_save(source, context, lambda encoder: MLPModel(encoder, mlp),
                         out_dir, name, seed)
