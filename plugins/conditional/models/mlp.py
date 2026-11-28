"""
A neural network for P(ampacity, phase | context): the learned counterpart of the table.

The network sees the same context fields as the table, looks each one up in its
own embedding and maps the concatenation through a small MLP to one logit per
outcome.  Where the table backs off along a fixed order of fields when a context
is rare, the network shares what it learns about a field value (say, a 3-phase
240 A previous sibling) across every context that value appears in.

Training minimises the cross-entropy, which is the held-out metric itself
(nats instead of bits), with early stopping on whole training feeders held back
for validation.  The test split is never seen during fitting.

Sampling draws from the softmax with the generator's numpy ``rng``
(:func:`~plugins.conditional.models.base.draw`), exactly like the table, so a
seed reproduces a sample whichever model produced it.  Building and fitting seed
torch from ``MLPConfig.seed`` inside a forked RNG, so they never change the
caller's global torch state (which the staged MDN lengths draw from).
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from plugins.conditional.config import ContextConfig, MLPConfig
from plugins.conditional.encoding import ContextEncoder, OutcomeVocabulary
from plugins.conditional.models.base import ConditionalModel, draw

#: rows per forward pass when scoring or sampling (memory only; the result is the same)
EVAL_BATCH = 65536


class _Network(nn.Module):
    """Context fields -> one logit per outcome: an embedding per field, then an MLP.

    radixes     number of values of each field (``encoder.radix``), in the column
                order of the input (``encoder.active``)
    n_outcomes  number of outputs, one logit per outcome
    """

    def __init__(self, radixes: list[int], n_outcomes: int, cfg: MLPConfig):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(radix, cfg.embedding_dim) for radix in radixes)

        layers = []
        in_dim = len(radixes) * cfg.embedding_dim
        for _ in range(cfg.n_layers):
            layers += [nn.Linear(in_dim, cfg.hidden_dim), nn.GELU(), nn.Dropout(cfg.dropout)]
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, n_outcomes))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [n_rows, n_fields] long -> logits [n_rows, n_outcomes]; no softmax."""
        columns = [embedding(x[:, i]) for i, embedding in enumerate(self.embeddings)]
        return self.body(torch.cat(columns, dim=-1))


def validation_rows(feeder: np.ndarray, fraction: float,
                    rng: np.random.Generator) -> np.ndarray | None:
    """Mask of the rows held back for early stopping: all rows of a share of the feeders.

    Whole feeders, never single rows: nodes of one feeder are correlated, so a
    row-wise split would validate on near-copies of the training rows.  Returns
    None when the share rounds to no feeder or to every feeder (tiny data).
    """
    feeders = np.unique(feeder)
    n_validation = int(round(fraction * len(feeders)))
    if n_validation == 0 or n_validation == len(feeders):
        return None
    held_back = rng.choice(feeders, n_validation, replace=False)
    return np.isin(feeder, held_back)


class MLPModel(ConditionalModel):
    """p(outcome | context) by a feedforward network on the context fields."""

    name = "mlp"

    def __init__(self, encoder: ContextEncoder, config: MLPConfig | None = None):
        super().__init__(encoder)
        self.config = config or MLPConfig()
        #: per epoch: train and validation bits per node, filled by fit
        self.history: list[dict] = []
        radixes = [encoder.radix[field] for field in encoder.active]
        with torch.random.fork_rng():                    # CPU and CUDA: manual_seed seeds both
            torch.manual_seed(self.config.seed)          # same config, same initial weights
            self.network = _Network(radixes, self.vocabulary.n_outcomes, self.config)

    # ── numpy fields -> network outputs ──────────────────────────────────────
    def _inputs(self, fields: dict[str, np.ndarray]) -> torch.Tensor:
        """The active fields as columns, in ``encoder.active`` order: [n_rows, n_fields] long."""
        columns = np.stack([np.asarray(fields[field]) for field in self.encoder.active], axis=1)
        return torch.as_tensor(columns, dtype=torch.long)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        """Logits [n_rows, n_outcomes] in evaluation mode (dropout off), in chunks."""
        self.network.eval()
        with torch.no_grad():
            if len(x) == 0:
                return torch.zeros((0, self.vocabulary.n_outcomes))
            return torch.cat([self.network(x[start:start + EVAL_BATCH])
                              for start in range(0, len(x), EVAL_BATCH)])

    def _log_probs(self, fields) -> np.ndarray:
        """Natural-log probabilities [n_rows, n_outcomes]; log_softmax, so never -inf."""
        logits = self._logits(self._inputs(fields)).double()
        return torch.log_softmax(logits, dim=-1).numpy()

    def _bits(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Mean bits per row of the outcomes ``y`` (the validation score)."""
        return float(F.cross_entropy(self._logits(x), y).item() / np.log(2))

    # ── fitting ──────────────────────────────────────────────────────────────
    def fit(self, fields, targets) -> "MLPModel":
        """Adam on the cross-entropy; keeps the weights of the best validation epoch.

        Rows with an unseen pair (outcome -1) are dropped, as for the joint table.
        Training stops after ``patience`` epochs without a better validation score,
        or after ``max_epochs``.  Without enough feeders to hold some back, the
        training rows themselves are the validation set.
        """
        cfg = self.config
        keep = targets["outcome"] >= 0
        x = self._inputs({field: values[keep] for field, values in fields.items()})
        y = torch.as_tensor(targets["outcome"][keep], dtype=torch.long)

        held_back = validation_rows(targets["feeder"][keep], cfg.validation_fraction,
                                    np.random.default_rng(cfg.seed))
        if held_back is None:
            x_train, y_train, x_val, y_val = x, y, x, y
        else:
            held_back = torch.as_tensor(held_back)
            x_train, y_train = x[~held_back], y[~held_back]
            x_val, y_val = x[held_back], y[held_back]

        with torch.random.fork_rng():
            torch.manual_seed(cfg.seed)                  # shuffling and dropout
            optimiser = torch.optim.Adam(self.network.parameters(), lr=cfg.lr,
                                         weight_decay=cfg.weight_decay)
            best_bits, best_weights, waited = self._bits(x_val, y_val), self._weights(), 0
            self.history = []
            for epoch in range(1, cfg.max_epochs + 1):
                train_bits = self._epoch(x_train, y_train, optimiser)
                val_bits = self._bits(x_val, y_val)
                self.history.append({"epoch": epoch, "train": train_bits, "validation": val_bits})
                if val_bits < best_bits:
                    best_bits, best_weights, waited = val_bits, self._weights(), 0
                else:
                    waited += 1
                    if waited >= cfg.patience:
                        break
        self.network.load_state_dict(best_weights)
        return self

    def _epoch(self, x: torch.Tensor, y: torch.Tensor, optimiser) -> float:
        """One pass over the rows in shuffled mini-batches; mean training bits (dropout on)."""
        self.network.train()
        order = torch.randperm(len(y))
        total = 0.0
        for start in range(0, len(order), self.config.batch_size):
            batch = order[start:start + self.config.batch_size]
            loss = F.cross_entropy(self.network(x[batch]), y[batch])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            total += loss.item() * len(batch)
        return float(total / len(order) / np.log(2))       # plain float: weights_only loads it

    def _weights(self) -> dict[str, torch.Tensor]:
        """A copy of the current weights (the state dict holds references, not copies)."""
        return {key: value.detach().clone() for key, value in self.network.state_dict().items()}

    # ── ConditionalModel ─────────────────────────────────────────────────────
    def log_prob(self, fields, targets) -> np.ndarray:
        if np.any(targets["outcome"] < 0):
            raise ValueError("outcome -1 is a pair never seen in training; "
                             "the network has no output for it")
        log_probs = self._log_probs(fields)
        return log_probs[np.arange(len(log_probs)), targets["outcome"]]

    def sample(self, fields, rng: np.random.Generator) -> np.ndarray:
        draws = draw(np.exp(self._log_probs(fields)), rng)
        return self.vocabulary.pairs[draws]

    def state(self) -> dict:
        return {"model": self.name,                               # which class rebuilds it
                "vocabulary": self.vocabulary.state(),
                "context": asdict(self.encoder.config),           # ContextConfig -> dict
                "mlp": asdict(self.config),                       # MLPConfig -> dict
                "weights": self._weights(),                       # tensors: weights_only loads them
                "history": self.history}

    @classmethod
    def from_state(cls, state: dict) -> "MLPModel":
        vocabulary = OutcomeVocabulary.from_state(state["vocabulary"])
        context = dict(state["context"])
        context["subtree_edges"] = tuple(context["subtree_edges"])    # frozen config: tuple
        encoder = ContextEncoder(vocabulary, ContextConfig(**context))
        model = cls(encoder, MLPConfig(**state["mlp"]))
        model.network.load_state_dict(state["weights"])
        model.history = list(state.get("history", []))
        return model
