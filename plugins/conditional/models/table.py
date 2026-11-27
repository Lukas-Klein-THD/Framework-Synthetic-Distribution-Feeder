"""
Hierarchical back-off tables: P(ampacity, phase | context) by counting.

TODO module docstring: what the model is, why it backs off along the field order
(coarsest = parent alone), what ``factorised`` isolates, same family as the
analysis notebook.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from plugins.conditional.config import ContextConfig, TableConfig
from plugins.conditional.encoding import N_PHASES, ContextEncoder, OutcomeVocabulary
from plugins.conditional.models.base import ConditionalModel

def _draw(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One class per row of 'p' [N, n_out], by inverse CDF"""
    c = p.cumsum(1)
    u = rng.random(len(p))[:, None] * c[:, -1:]
    return np.minimum((u > c).sum(1), p.shape[1] - 1)

def count_table(context: np.ndarray, y: np.ndarray, n_out: int):
    """Count outcomes per context.

    context  [N] integer context id per row
    y        [N] outcome per row, in 0..n_out-1
    returns  (ids, counts): ids [C] the sorted distinct contexts,
             counts [C, n_out] how often each outcome occurred in each
    """
    ids, row_positions = np.unique(context, return_inverse=True)

    counts = np.zeros((len(ids), n_out))
    np.add.at(counts, (row_positions.ravel(), y), 1)

    return ids, counts

def lookup_counts(ids: np.ndarray, counts: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Count rows for the contexts ``context`` [N]; zeros where a context is unseen.

    ids, counts  a table from count_table
    returns      [N, n_out]
    """
    # 1. build a dict: id -> row number in counts
    id_to_row = {id_: row for row, id_ in enumerate(ids)}

    # 2. make an output of zeros, shape [len(context), counts.shape[1]]
    out = np.zeros((len(context), counts.shape[1]))

    # 3. for each position k and id c in context:
    #        if c is in the dict: copy that row of counts into output row k
    for position, context_id in enumerate(context):
        if context_id in id_to_row:
            row = id_to_row[context_id]
            out[position] = counts[row]
    # 4. return the output
    return out

def marginal(outcomes: np.ndarray, n_outcomes: int) -> np.ndarray:
    """How common each outcome is over all training nodes, [n_outcomes]; never 0.
    Formula: (count of each outcome + 0.5) / (number of nodes + 0.5 · n_outcomes)"""
    counts = np.bincount(outcomes, minlength=n_outcomes)
    return (counts + 0.5) / (len(outcomes) + 0.5 * n_outcomes)
    
def smooth(context_counts: np.ndarray, coarser_probs: np.ndarray, beta: float) -> np.ndarray:
    """Mix the counts of a context with the estimate from the coarser level.
    Formula: (context_counts + beta · coarser_probs) / (total count of the context + beta)

    context_counts  [n_rows, n_outcomes]  counts from lookup_counts
    coarser_probs   [n_rows, n_outcomes]  probabilities from the level before
    beta            how many imaginary observations the coarser estimate is worth
    returns         [n_rows, n_outcomes]  probabilities, each row sums to 1
    """
    total_counts = context_counts.sum(1, keepdims=True)
    return (context_counts + beta * coarser_probs) / (total_counts + beta)

def backoff_probs(marginal_probs: np.ndarray, level_tables: list, level_query_ids: list,
                  beta: float) -> np.ndarray:
    """Probabilities after smoothing through every level, coarse to fine.

    marginal_probs   [n_outcomes]                          from marginal()
    level_tables     one (table_ids, table_counts) per level, from count_table()
    level_query_ids  one [n_rows] array of context ids per level (the same rows each time)
    returns          [n_rows, n_outcomes]
    """
    n_rows = len(level_query_ids[0])
    probs = np.tile(marginal_probs, (n_rows, 1))
    for (table_ids, table_counts), query_ids in zip(level_tables, level_query_ids):
        context_counts = lookup_counts(table_ids, table_counts, query_ids)
        probs = smooth(context_counts, probs, beta)
    return probs


class _Backoff:
    """p(outcome | context) for one target: the marginal and one table per level."""

    def __init__(self, n_outcomes: int, beta: float):
        self.n_outcomes = n_outcomes
        self.beta = beta
        self.marginal_probs = None     # set by fit
        self.level_tables = []         # set by fit: one (table_ids, table_counts) per level

    def fit(self, encoder, fields, outcomes) -> "_Backoff":
        # 1. marginal of the outcomes
        self.marginal_probs = marginal(outcomes, self.n_outcomes)
        # 2. for each names in encoder.levels():
        #        context_ids = encoder.index(fields, names)
        #        append count_table(context_ids, outcomes, self.n_outcomes)
        self.level_tables = []
        for names in encoder.levels():
            context_ids = encoder.index(fields, names)
            table = count_table(context_ids, outcomes, self.n_outcomes)
            self.level_tables.append(table)
        # 3. return self
        return self

    def probs(self, encoder, fields) -> np.ndarray:
        # 1. for each names in encoder.levels(): query ids = encoder.index(fields, names)
        level_query_ids = []
        for names in encoder.levels():
            level_query_ids.append(encoder.index(fields, names))
        # 2. return backoff_probs(self.marginal_probs, self.level_tables, those ids, self.beta)
        return backoff_probs(self.marginal_probs, self.level_tables, level_query_ids, self.beta)

    def state(self) -> dict:
        """Plain Python types only, so ``torch.load(weights_only=True)`` accepts it."""
        return {"n_outcomes": self.n_outcomes,
                "beta": self.beta,
                "marginal_probs": self.marginal_probs.tolist(),
                "level_tables": [[table_ids.tolist(), table_counts.tolist()]
                                 for table_ids, table_counts in self.level_tables]}

    @classmethod
    def from_state(cls, state: dict) -> "_Backoff":
        backoff = cls(state["n_outcomes"], state["beta"])
        backoff.marginal_probs = np.array(state["marginal_probs"])
        backoff.level_tables = [(np.array(table_ids, dtype=np.int64), np.array(table_counts))
                                for table_ids, table_counts in state["level_tables"]]
        return backoff


class BackoffTable(ConditionalModel):
    """p(outcome | context) by hierarchical back-off over the context fields."""

    name = "table"

    def __init__(self, encoder: ContextEncoder, config: TableConfig | None = None):
        super().__init__(encoder)              # stores the encoder (base.py)
        self.config = config or TableConfig()
        self.tables = {}                       # target name -> _Backoff, set by fit

    def _targets(self) -> dict[str, int]:
        """Target name -> number of classes."""
        if self.config.factorised:
            return {"ampacity_class": self.vocabulary.n_ampacity, "phase": N_PHASES}
        return {"outcome": self.vocabulary.n_outcomes}

    def _check_fitted(self) -> None:
        if not self.tables:                    # empty dict = not fitted
            raise RuntimeError("BackoffTable is not fitted; call fit() or from_state()")

    def fit(self, fields, targets) -> "BackoffTable":
        if self.config.factorised:
            keep = np.ones(len(targets["outcome"]), dtype=bool)      # every row is usable
        else:
            keep = targets["outcome"] >= 0                           # joint: only known pairs
        fields = {name: values[keep] for name, values in fields.items()}
        targets = {name: values[keep] for name, values in targets.items()}

        self.tables = {}
        for name, n_classes in self._targets().items():
            self.tables[name] = _Backoff(n_classes, self.config.beta).fit(
                self.encoder, fields, targets[name])
        return self

    def log_prob(self, fields, targets) -> np.ndarray:
        self._check_fitted()
        if not self.config.factorised and np.any(targets["outcome"] < 0):
            raise ValueError("outcome -1 is a pair never seen in training; "
                             "a joint table cannot score it")
        n_rows = len(targets["outcome"])
        total = np.zeros(n_rows)
        for name, table in self.tables.items():
            p = table.probs(self.encoder, fields)
            total += np.log(p[np.arange(n_rows), targets[name]])
        return total

    def sample(self, fields, rng: np.random.Generator) -> np.ndarray:
        self._check_fitted()
        draws = {}
        for name, table in self.tables.items():
            draws[name] = _draw(table.probs(self.encoder, fields), rng)

        if self.config.factorised:
            return draws["ampacity_class"] * N_PHASES + draws["phase"]
        return self.vocabulary.pairs[draws["outcome"]]

    def state(self) -> dict:
        self._check_fitted()
        return {"model": self.name,                               # which class rebuilds it
                "vocabulary": self.vocabulary.state(),
                "context": asdict(self.encoder.config),           # ContextConfig -> dict
                "table": asdict(self.config),                     # TableConfig -> dict
                "tables": {name: backoff.state() for name, backoff in self.tables.items()}}

    @classmethod
    def from_state(cls, state: dict) -> "BackoffTable":
        vocabulary = OutcomeVocabulary.from_state(state["vocabulary"])
        context = dict(state["context"])
        context["subtree_edges"] = tuple(context["subtree_edges"])    # frozen config: tuple
        encoder = ContextEncoder(vocabulary, ContextConfig(**context))
        model = cls(encoder, TableConfig(**state["table"]))
        model.tables = {name: _Backoff.from_state(s) for name, s in state["tables"].items()}
        return model
