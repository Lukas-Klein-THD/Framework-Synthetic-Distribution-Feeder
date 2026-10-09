"""
Compare generators under one evaluation.

    python examples/compare.py --n 512 --device cuda

Every model is scored against the same real reference set with the same metric
instances (so the random GIN embedding is identical for all).  The floor is a
second, disjoint real set of the same size, scored exactly like a model.  That
keeps sample sizes equal, which precision, recall and the Frechet distance are
sensitive to.

Electrical checks (constraints, power flow) assume a tree, so they run on the
generated feeders that are trees; the count is printed to the log.  Results go to ``results/<date>_model_comparison/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridgen import Evaluator                                              # noqa: E402
from gridgen.evaluation.base import Direction                              # noqa: E402
from plugins.distribution import (                                         # noqa: E402
    AmpacityTapering, AttributeEncoder, PhaseSubset, TensorFeederSource,
    attributed_metrics, structural_metrics)
from plugins.distribution import config as shared                          # noqa: E402
from plugins.distribution.metrics import (                                 # noqa: E402
    is_radial_tree)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_LINECODES = os.path.join(REPO, "..", "paper-implementation", "data", "dss_data",
                                 "GSO_Base_Network", "Urban-Suburban", "Linecodes.dss")


def tree_shape(feeders) -> dict[str, float]:
    """Plain topology statistics of the valid trees, for reading the metrics."""
    leaf, branch, max_out, depth, diam = [], [], [], [], []
    for f in feeders:
        a = np.asarray(f.adjacency) != 0
        out = a.sum(1)
        leaf.append((out == 0).mean()); branch.append((out >= 2).mean())
        max_out.append(out.max())
        parents = a.argmax(0)
        d = np.zeros(len(out))
        for j in range(1, len(out)):
            d[j] = d[parents[j]] + 1
        depth.append(d.max())
        # diameter of a tree: two BFS passes on the undirected graph
        u = a | a.T

        def far(s):
            dist = {s: 0}; frontier = [s]
            while frontier:
                nxt = []
                for x in frontier:
                    for y in np.flatnonzero(u[x]):
                        if int(y) not in dist:
                            dist[int(y)] = dist[x] + 1; nxt.append(int(y))
                frontier = nxt
            node = max(dist, key=dist.get)
            return node, dist[node]
        diam.append(far(far(0)[0])[1])
    return {"leaf_share": float(np.mean(leaf)), "branching_share": float(np.mean(branch)),
            "max_out_degree": float(np.mean(max_out)), "max_depth": float(np.mean(depth)),
            "diameter": float(np.mean(diam))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--linecodes", default=os.environ.get("GRIDGEN_LINECODES", DEFAULT_LINECODES))
    p.add_argument("--powerflow-feeders", type=int, default=64)
    p.add_argument("--staged-dir", default=os.path.join(shared.CHECKPOINT_ROOT, "staged"))
    p.add_argument("--improved", default=os.path.join(shared.CHECKPOINT_ROOT, "feedergan_improved",
                                                      "feedergan_improved_final.pt"))
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from plugins.feedergan_improved import FeederGANImproved
    from plugins.staged import build_staged

    # ── real data: reference set and an equal-sized, disjoint floor set ───────
    source = TensorFeederSource()
    pool = source.sample(2 * args.n, seed=args.seed)
    reference, floor = pool[:args.n], pool[args.n:]
    print(f"{source}: reference {len(reference)}, floor {len(floor)}")

    # ── generators ───────────────────────────────────────────────────────────
    models = {
        "staged": build_staged(os.path.join(args.staged_dir, "topology.pt"),
                               os.path.join(args.staged_dir, "attributes.pt"),
                               device=args.device),
        "feedergan-improved": FeederGANImproved(args.improved, device=args.device),
    }
    sets = {"real (floor)": floor}
    for name, gen in models.items():
        torch.manual_seed(args.seed)
        sets[name] = gen.sample(args.n)
        print(f"sampled {len(sets[name])} feeders from {name}")

    # ── metrics: one instance of each, shared by every set ───────────────────
    encoder = AttributeEncoder.fit(reference)
    metrics = structural_metrics(seed=args.seed) + attributed_metrics(encoder, seed=args.seed)
    powerflow = None
    if args.linecodes and os.path.isfile(args.linecodes):
        from plugins.distribution.powerflow import PowerFlowFeasibility
        powerflow = PowerFlowFeasibility(linecodes=args.linecodes,
                                         max_feeders=args.powerflow_feeders)
    else:
        print("power flow skipped: no Linecodes.dss")
    constraints = [PhaseSubset(), AmpacityTapering()]

    results: dict[str, dict] = {}
    directions: dict[str, str] = {}
    for name, feeders in sets.items():
        card = Evaluator(metrics).run(reference, feeders, with_reference=False,
                                      title=name, seed=args.seed)
        row = dict(card.scores)
        directions.update({k: v.name for k, v in card.directions.items()})
        trees = [f for f in feeders if is_radial_tree(f.adjacency)]
        print(f"{name}: {len(trees)}/{len(feeders)} trees used for the electrical checks")
        for c in constraints:
            v = c.validate(trees)
            row[f"{c.name}.edges"], row[f"{c.name}.feeders"] = v.element_rate, v.feeder_rate
            directions[f"{c.name}.edges"] = directions[f"{c.name}.feeders"] = "HIGHER_IS_BETTER"
        row.update({f"shape.{k}": v for k, v in tree_shape(trees).items()})
        if powerflow is not None:
            pf = powerflow.compute(reference, trees)
            row.update({f"powerflow.{k}": v for k, v in pf.items()})
            directions.update({f"powerflow.{k}": "HIGHER_IS_BETTER" for k in pf})
        results[name] = row
        print(f"scored {name}")
    real_row = {f"{c.name}.{k}": getattr(c.calibrate(reference), r)
                for c in constraints for k, r in (("edges", "element_rate"),
                                                   ("feeders", "feeder_rate"))}
    real_row.update({f"shape.{k}": v for k, v in tree_shape(reference).items()})
    results["real (reference)"] = real_row

    # ── write ────────────────────────────────────────────────────────────────
    out = args.out or os.path.join(REPO, "results",
                                   f"{dt.date.today():%Y-%m-%d}_model_comparison")
    os.makedirs(out, exist_ok=True)
    meta = {"n": args.n, "seed": args.seed, "linecodes": args.linecodes,
            "powerflow_feeders": args.powerflow_feeders,
            "powerflow_config": vars(powerflow.cfg) if powerflow else None,
            "checkpoints": {"staged": args.staged_dir, "feedergan-improved": args.improved}}
    with open(os.path.join(out, "scores.json"), "w") as f:
        json.dump({"meta": meta, "directions": directions, "results": results}, f, indent=2,
                  default=float)

    columns = ["real (reference)", "real (floor)", "staged", "feedergan-improved"]
    keys = sorted({k for r in results.values() for k in r})
    arrow = {"LOWER_IS_BETTER": "lower", "HIGHER_IS_BETTER": "higher"}
    lines = ["| metric | better | " + " | ".join(columns) + " |",
             "|---|---|" + "---|" * len(columns)]
    for k in keys:
        cells = []
        for c in columns:
            v = results.get(c, {}).get(k)
            cells.append("-" if v is None else (f"{v:.4g}" if isinstance(v, float) else str(v)))
        lines.append(f"| {k} | {arrow.get(directions.get(k, ''), '')} | " + " | ".join(cells) + " |")
    with open(os.path.join(out, "table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
