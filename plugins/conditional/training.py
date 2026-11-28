"""
Fitting the conditional models on training data and saving them as checkpoints.
...
"""


from __future__ import annotations
import os

import numpy as np
import torch

from plugins.conditional.encoding import ContextEncoder, OutcomeVocabulary
from plugins.conditional.models.table import BackoffTable
from plugins.conditional import config
from plugins.conditional.config import ContextConfig, TableConfig


def fit_table(source, context: ContextConfig | None = None, table: TableConfig | None = None,
              out_dir: str = config.CHECKPOINT_DIR, name: str = "table",
              seed: int = 0) -> tuple[BackoffTable, str]:
    """Fit a back-off table on ``source`` and save it as ``<out_dir>/<name>.pt``.

    Returns the fitted model and the checkpoint path.
    """
    # 1. Load data from the source into a list of feeders
    feeders = list(source)
    # 2. Define the outcome vocabulary and fit it on the training feeders
    vocabulary = OutcomeVocabulary().fit(feeders)
    # 3. Define the context encoder
    encoder = ContextEncoder(vocabulary, context)
    # 4. Encode the feeders
    fields, targets = encoder.encode_feeders(feeders, np.random.default_rng(seed))
    # 5. Fit the back-off table model
    model = BackoffTable(encoder, table).fit(fields, targets)
    # 6. Create the output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.pt")
    # 7. Save the model state to the checkpoint path
    torch.save(model.state(), path)
    # 8. Return the fitted model and the checkpoint path
    return model, path