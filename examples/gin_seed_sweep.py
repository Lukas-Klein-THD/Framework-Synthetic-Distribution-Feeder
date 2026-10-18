"""
Embedding metrics over several random GIN initialisations.

    python examples/gin_seed_sweep.py --device cuda --out <results dir> --seeds 0 1 2 3 4

The random-GIN embedding is untrained, so its initialisation is a source of
variance (Thompson et al., 2022).  This script draws the feeder sets of Tables I
and III exactly as ``full_results.py`` does (same real sets, sampling seeds and
order), keeps them fixed, and scores them under each GIN seed.  Only the
embedding changes between seeds, so seed 0 reproduces the published numbers.

Writes ``<out>/logs/gin_seed_sweep.log`` and ``<out>/gin_seeds.json`` (per-seed
values, mean and sample standard deviation).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gridgen.core.feeder import Feeder                                     # noqa: E402
from plugins.distribution import attributed_metrics, structural_metrics    # noqa: E402
from plugins.distribution.metrics import realism_score                    # noqa: E402
from plugins.distribution import config as shared                          # noqa: E402
from full_results import (                                                 # noqa: E402
    Context, _Tee, _echo, evaluate, generate_attributes, random_trees, repair, with_attributes)

STRUCT = ("gin.prdc.precision", "gin.prdc.recall", "gin.fid.value")
ATTR = ("gin-attributed.prdc.precision", "gin-attributed.prdc.recall",
        "gin-attributed.fid.value")


#: realism score per metric family: (precision key, recall key) -> score key
SCORES = {"score.structure": ("gin.prdc.precision", "gin.prdc.recall"),
          "score.attributes": ("gin-attributed.prdc.precision", "gin-attributed.prdc.recall")}


def add_realism_scores(summary: dict) -> dict:
    """Add the 0-10 realism score to every row, computed per embedding seed.

    Each seed's score is taken against that seed's real-vs-real row, then
    averaged, so the score carries the same mean +- std as the other metrics.
    Works on a loaded ``gin_seeds.json`` summary as well; existing scores are
    recomputed.
    """
    for rows in summary.values():
        ref = rows.get("real vs real")
        if ref is None:
            continue
        for row in rows.values():
            for key, (pk, rk) in SCORES.items():
                if pk not in row or pk not in ref:
                    continue
                values = [realism_score(p, r, rp, rr) for p, r, rp, rr in zip(
                    row[pk]["values"], row[rk]["values"],
                    ref[pk]["values"], ref[rk]["values"])]
                row[key] = {"values": values, "mean": float(np.mean(values)),
                            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
            # combined realism: arithmetic mean of the two families, per seed, so
            # a failure in one family halves the score instead of erasing it
            if all(k in row for k in SCORES):
                values = [float(np.mean(v)) for v in zip(*(row[k]["values"] for k in SCORES))]
                row["score.realism"] = {
                    "values": values, "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}
    return summary


def table_sets(ctx):
    """Table I (n = 512) and Table III (n = 256) sets, as full_results.py draws them."""
    m = shared.NUM_NODES
    table1 = {
        "real vs real": ctx.floor,
        "staged": ctx.samples("staged"),
        "feedergan-improved": ctx.samples("feedergan-improved"),
        "uniform random tree": [Feeder(a) for a in random_trees(ctx.n, m, "uniform", ctx.seed)],
        # the same feeders after post-hoc phase repair (topology unchanged)
        "staged + phase repair": repair(ctx.samples("staged")),
        "feedergan-improved + phase repair": repair(ctx.samples("feedergan-improved")),
    }
    n3 = min(256, ctx.n)
    reference3 = ctx.reference[:n3]
    marginals = ctx.marginals if ctx.heldout else reference3
    torch.manual_seed(ctx.seed)
    gan = ctx.staged.topology.sample_topology(n3)
    rnd = random_trees(n3, m, "uniform", ctx.seed)
    table3 = {
        "real vs real": ctx.floor[:n3],
        "GAN + MDN (staged)": with_attributes(gan, generate_attributes(ctx.mdn, gan, seed=ctx.seed)),
        "GAN + i.i.d. echo": with_attributes(gan, _echo(ctx, gan, marginals, ctx.seed)),
        "random tree + MDN": with_attributes(rnd, generate_attributes(ctx.mdn, rnd, seed=ctx.seed)),
        "random tree + echo": with_attributes(rnd, _echo(ctx, rnd, marginals, ctx.seed)),
        "feedergan-improved": ctx.samples("feedergan-improved")[:n3],
    }
    return table1, (reference3, table3)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--all-data", action="store_true",
                   help="earlier protocol: real sets drawn from every graph")
    p.add_argument("--seed", type=int, default=0, help="sampling seed (fixed)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    args.custom_approach, args.linecodes = None, None

    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    log_path = os.path.join(args.out, "logs", "gin_seed_sweep.log")
    with open(log_path, "w", encoding="utf-8") as log, \
            contextlib.redirect_stdout(_Tee(sys.__stdout__, log)):
        ctx = Context(args)
        table1, (reference3, table3) = table_sets(ctx)
        per_seed: dict[str, dict[str, dict[str, list[float]]]] = {"table1": {}, "table3": {}}

        for gin_seed in args.seeds:
            print(f"--- GIN seed {gin_seed} ---")
            m1 = structural_metrics(seed=gin_seed) + attributed_metrics(ctx.encoder, seed=gin_seed)
            for name, feeders in table1.items():
                keys = STRUCT if name == "uniform random tree" else STRUCT + ATTR
                metrics = m1 if name != "uniform random tree" else structural_metrics(seed=gin_seed)
                s = evaluate(metrics, ctx.reference, feeders, ctx.seed, name)
                row = per_seed["table1"].setdefault(name, {})
                for k in keys:
                    row.setdefault(k, []).append(s[k])
                print(f"table1 {name:<34}" + "".join(f"{s[k]:>10.4g}" for k in keys))
            # each stage is read off its own metric family: structure for the
            # topology stage, attributes for the attribute stage
            m3 = structural_metrics(seed=gin_seed) + attributed_metrics(ctx.encoder, seed=gin_seed)
            for name, feeders in table3.items():
                s = evaluate(m3, reference3, feeders, ctx.seed, name)
                row = per_seed["table3"].setdefault(name, {})
                for k in STRUCT + ATTR:
                    row.setdefault(k, []).append(s[k])
                print(f"table3 {name:<22}" + "".join(f"{s[k]:>10.4g}" for k in STRUCT + ATTR))

        summary = {t: {name: {k: {"values": v, "mean": float(np.mean(v)),
                                  "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                              for k, v in row.items()}
                       for name, row in rows.items()}
                   for t, rows in per_seed.items()}
        add_realism_scores(summary)
        print(f"\n=== mean +- std over GIN seeds {args.seeds} ===")
        for t, rows in summary.items():
            print(f"\n{t}")
            for name, row in rows.items():
                print(f"  {name:<22}" + "  ".join(
                    f"{k.split('.')[-2] if 'prdc' in k else 'FD'}"
                    f"{'(attr)' if 'attributed' in k else ''} "
                    f"{v['mean']:.3g}+-{v['std']:.2g}" for k, v in row.items()))
    with open(os.path.join(args.out, "gin_seeds.json"), "w") as f:
        json.dump({"gin_seeds": args.seeds, "sampling_seed": args.seed, "n": args.n,
                   "summary": summary}, f, indent=2)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
