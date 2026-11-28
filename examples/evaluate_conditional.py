"""
The WP1 ladder of conditional tables, scored on the paper's three dimensions.

    python examples/train.py --model conditional-table
    python examples/evaluate_conditional.py --out <results dir>

Run both with ``GRIDGEN_CHECKPOINT_DIR=checkpoints/heldout`` so the tables and
the staged model come from the same held-out training split.

The held-out bits score one node at a time with the true context.  Here whole
feeders are generated, so every node is conditioned on the model's own earlier
draws, and they are scored as a set, like the staged model in ``full_results.py``:

    topologies   the real comparison-half feeders, kept as they are; only
                 ampacity and phase are generated (stage 2 on real topology)
    reference    the other test half; the metric encoding is fitted on training
    lengths      the staged MDN's, one draw per sampling seed, shared by every
                 row, so the rows differ only in ampacity and phase

Rows: the real comparison feeders; real ampacity and phase with the MDN
lengths (the ceiling for every generated row); the staged MDN; the ladder rungs.

The three dimensions, as in ``summary_scores.py``:

    realism       attributed realism only: 10 * F1(P, R) / F1(real vs real), per
                  GIN seed against that seed's real row.  The topology is real, so
                  structural realism would be 10 for every row and is left out.
    physical      10 * mean(phase-valid edges, phase-valid feeders)
    operational   OpenDSS on the first ``--pf-n`` feeders of every sampling seed,
                  10 * E * mean(V, W, U) at ``--pf-kw`` per node (the paper uses
                  one draw; the seeds here give the score a spread)
    overall       the mean of the three

These describe attribute quality on real topology; the paper's full-pipeline
rows also include generated topology, so the numbers are not interchangeable.
Spreads: ``sd sampling`` over sampling seeds (each value first averaged over the
GIN seeds), ``sd GIN`` over GIN seeds (averaged over sampling seeds).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from full_results import (AMP_KEYS, REPO, Context, _Tee, evaluate,            # noqa: E402
                          generate_attributes, persistence, tv, validity, values,
                          with_attributes)
from plugins.conditional import config                                        # noqa: E402
from plugins.conditional.generator import ConditionalAttributes               # noqa: E402
from plugins.conditional.models import load_model                             # noqa: E402
from plugins.distribution.metrics import (operational_score, physical_score,  # noqa: E402
                                          realism_score)
from plugins.graph_metrics import attributed_metrics                          # noqa: E402

REAL, CEILING, STAGED = "real (comparison)", "real A, P + MDN lengths", "staged MDN"
#: metrics that depend on the GIN initialisation
GIN_KEYS = ("precision", "recall", "coverage", "density", "fd")

#: (key, column header, format) of the per-metric table
COLUMNS = [("precision", "prec", ".3f"), ("recall", "rec", ".3f"), ("coverage", "cov", ".3f"),
           ("fd", "FD", ".0f"), ("realism", "real.", ".2f"),
           ("ampacity_tv", "amp TV", ".3f"), ("phase_tv", "ph TV", ".3f"),
           ("ampacity_persistence", "amp pers", ".3f"), ("phase_persistence", "ph pers", ".3f"),
           ("phase_valid_edges", "ph edges", ".4f"), ("phase_valid_feeders", "ph feed", ".3f"),
           ("taper_edges", "taper", ".3f"), ("physical", "phys.", ".2f")]


# ── realism and physical consistency ─────────────────────────────────────────
def gin_metrics(ctx, metrics_by_gin, feeders) -> dict:
    """Attributed precision / recall / coverage / density / FD, per GIN seed."""
    out = {}
    for gin_seed, metrics in metrics_by_gin.items():
        s = evaluate(metrics, ctx.reference, feeders, ctx.seed, "")
        out[gin_seed] = {k.split(".")[-1] if "prdc" in k else "fd": float(s[k])
                         for k in AMP_KEYS}
    return out


def plain_metrics(ctx, feeders) -> dict:
    """The metrics without an embedding: marginals, persistence, validity."""
    ref = ctx.reference
    v = validity(feeders)
    out = {"ampacity_tv": tv(values(ref, "ampacity"), values(feeders, "ampacity"),
                             ctx.encoder.ampacities),
           "phase_tv": tv(values(ref, "phase"), values(feeders, "phase"), range(7)),
           "ampacity_persistence": persistence(feeders, "ampacity"),
           "phase_persistence": persistence(feeders, "phase"),
           "phase_valid_edges": v["phase-subset.edges"],
           "phase_valid_feeders": v["phase-subset.feeders"],
           "taper_edges": v["ampacity-tapering.edges"]}
    out["physical"] = physical_score(out["phase_valid_edges"], out["phase_valid_feeders"])
    return {k: float(x) for k, x in out.items()}


def generated_sets(ctx, models, adjacencies, seed) -> dict:
    """Every generated row for one sampling seed; all share the MDN lengths of that seed."""
    mdn = generate_attributes(ctx.mdn, adjacencies, seed=seed)
    real = [dict(m, ampacity=np.asarray(f.attributes["ampacity"]),
                 phase=np.asarray(f.attributes["phase"])) for m, f in zip(mdn, ctx.floor)]
    sets = {STAGED: with_attributes(adjacencies, mdn),
            CEILING: with_attributes(adjacencies, real)}
    for rung, model in models.items():
        generator = ConditionalAttributes(model, lengths=ctx.mdn)
        attrs = generator.generate(adjacencies, rng=np.random.default_rng(1000 + seed))
        # the generator draws its own MDN lengths; use the shared ones instead
        attrs = [dict(a, length=m["length"]) for a, m in zip(attrs, mdn)]
        sets[rung] = with_attributes(adjacencies, attrs)
    for name, feeders in sets.items():                  # only A and P may differ
        assert all(np.array_equal(f.attributes["length"], g.attributes["length"])
                   for f, g in zip(feeders, sets[STAGED])), name
    return sets


def over_seeds(matrix) -> dict:
    """[sampling seed, GIN seed] -> mean, sd over sampling seeds, sd over GIN seeds."""
    m = np.asarray(matrix, float)
    per_sampling = m.mean(axis=1)
    return {"mean": float(m.mean()),
            "sd_sampling": float(per_sampling.std(ddof=1)) if m.shape[0] > 1 else 0.0,
            "sd_gin": float(m.std(axis=1, ddof=1).mean()) if m.shape[1] > 1 else 0.0}


def summarise(rows: list[dict], real_gin: dict) -> dict:
    """One set's metrics over sampling seeds; realism per GIN seed against real."""
    gin_seeds = list(real_gin)
    out = {}
    for k in GIN_KEYS:
        out[k] = over_seeds([[r["gin"][e][k] for e in gin_seeds] for r in rows])
    out["realism"] = over_seeds([[realism_score(r["gin"][e]["precision"], r["gin"][e]["recall"],
                                                real_gin[e]["precision"], real_gin[e]["recall"])
                                  for e in gin_seeds] for r in rows])
    for k in rows[0]["plain"]:
        x = [r["plain"][k] for r in rows]
        out[k] = {"mean": float(np.mean(x)),
                  "sd_sampling": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0}
    return out


# ── operational performance ──────────────────────────────────────────────────
#: (key, column header, format) of the power-flow table
FLOW_COLUMNS = [("energised", "energised", ".3f"), ("vmin", "Vmin", ".4f"),
                ("ansi_a", "ANSI-A", ".3f"), ("unbalance_pct", "unbal%", ".2f"),
                ("unbalance_ok", "unb<=3%", ".3f"), ("loading", "loading", ".3f"),
                ("overload", "overload", ".3f"), ("operational", "oper.", ".2f")]


def power_flow(arms, args) -> dict:
    """OpenDSS on every arm, as ``full_results.power_flow`` (same harness and settings).

    Returns ``{"<kW> kW | <arm>": row}`` for the first ``--pf-n`` feeders of each arm.
    """
    sys.path.append(os.path.abspath(args.custom_approach))
    from postprocessing.powerflow import LinecodeLibrary, run_feeder

    lib = LinecodeLibrary(args.linecodes)

    def matrix(f):
        return np.stack([f.attributes["length"], f.attributes["ampacity"],
                         f.attributes["phase"]], axis=1).astype(np.float32)

    rows = {}
    for kw in args.pf_loads:
        for name, feeders in arms.items():
            res = [run_feeder(np.asarray(f.adjacency), matrix(f), lib, kw_per_node=kw,
                              length_units="km", source_pu=args.source_pu)
                   for f in feeders[:args.pf_n]]

            def m(key):
                return float(np.nanmean([r.get(key, np.nan) for r in res]))
            row = {"converged": float(np.mean([r["converged"] for r in res])),
                   "energised": m("energised_frac"), "vmin": m("vmin"),
                   "ansi_a": m("in_band_frac"), "ansi_b": m("rangeB_frac"),
                   "unbalance_pct": m("unbalance_mean"), "unbalance_ok": m("unbalance_ok_frac"),
                   "loading": m("loading_mean"), "overload": m("overload_frac")}
            row["operational"] = operational_score(row["energised"], row["ansi_a"],
                                                   1.0 - row["overload"], row["unbalance_ok"])
            rows[f"{kw:g} kW | {name}"] = row
    return rows


def flow_summary(per_seed_flow: list[dict]) -> dict:
    """Mean and sd over sampling seeds of every power-flow number."""
    out = {}
    for key in per_seed_flow[0]:
        rows = [flow[key] for flow in per_seed_flow]
        out[key] = {k: {"mean": float(np.mean([r[k] for r in rows])),
                        "sd_sampling": float(np.std([r[k] for r in rows], ddof=1))
                        if len(rows) > 1 else 0.0}
                    for k in rows[0]}
    return out


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=256, help="feeders per set")
    p.add_argument("--seed", type=int, default=0, help="seed of the real sets")
    p.add_argument("--gin-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="GIN initialisations of the attributed metrics")
    p.add_argument("--gen-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="sampling seeds of the generated attributes")
    p.add_argument("--rungs", nargs="+", default=list(config.LADDER),
                   choices=list(config.LADDER))
    p.add_argument("--checkpoints", default=config.CHECKPOINT_DIR,
                   help="folder with <rung>.pt from examples/train.py --model conditional-table")
    p.add_argument("--no-power-flow", action="store_true", help="skip the operational dimension")
    p.add_argument("--pf-n", type=int, default=64, help="feeders per arm in the power flow")
    p.add_argument("--pf-loads", type=float, nargs="+", default=[10.0, 50.0, 100.0],
                   help="kW per node")
    p.add_argument("--pf-kw", type=float, default=50.0, help="load of the operational score")
    p.add_argument("--source-pu", type=float, default=1.03)
    p.add_argument("--custom-approach", default=os.path.join(REPO, "..", "custom-approach"))
    p.add_argument("--linecodes", default=os.path.join(
        REPO, "..", "paper-implementation", "data", "dss_data", "GSO_Base_Network",
        "Urban-Suburban", "Linecodes.dss"))
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if not args.no_power_flow and args.pf_kw not in args.pf_loads:
        p.error("--pf-kw must be one of --pf-loads")
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)

    with open(os.path.join(args.out, "logs", "evaluate_conditional.log"), "w",
              encoding="utf-8") as log, contextlib.redirect_stdout(_Tee(sys.__stdout__, log)):
        ctx = Context(args)
        adjacencies = [np.asarray(f.adjacency) for f in ctx.floor]
        metrics_by_gin = {e: attributed_metrics(ctx.encoder, seed=e) for e in args.gin_seeds}
        models = {rung: load_model(os.path.join(args.checkpoints, f"{rung}.pt"))
                  for rung in args.rungs}
        print(f"{len(adjacencies)} comparison topologies, reference {len(ctx.reference)}, "
              f"sampling seeds {args.gen_seeds}, GIN seeds {args.gin_seeds}")
        print(f"tables from {args.checkpoints}\n")

        # ── realism and physical consistency ─────────────────────────────────
        per_seed = {REAL: [{"gin": gin_metrics(ctx, metrics_by_gin, ctx.floor),
                            "plain": plain_metrics(ctx, ctx.floor)}]}
        flow_arms = {}                                   # sampling seed -> power-flow arms
        for g in args.gen_seeds:
            sets = generated_sets(ctx, models, adjacencies, g)
            flow_arms[g] = {name: feeders[:args.pf_n] for name, feeders in sets.items()}
            for name, feeders in sets.items():
                per_seed.setdefault(name, []).append(
                    {"gin": gin_metrics(ctx, metrics_by_gin, feeders),
                     "plain": plain_metrics(ctx, feeders)})
            print(f"  sampling seed {g} scored", flush=True)

        real_gin = per_seed[REAL][0]["gin"]
        results = {name: summarise(rows, real_gin) for name, rows in per_seed.items()}

        print("\nmean over sampling and GIN seeds\n")
        print(f"{'set':<26}" + "".join(f"{h:>10}" for _, h, _ in COLUMNS))
        for name, r in results.items():
            print(f"{name:<26}" + "".join(f"{r[k]['mean']:>10{f}}" for k, _, f in COLUMNS))
        print("\nprecision sd over sampling seeds / over GIN seeds: " + ", ".join(
            f"{name} {r['precision']['sd_sampling']:.3f}/{r['precision']['sd_gin']:.3f}"
            for name, r in results.items()))

        # ── operational performance ──────────────────────────────────────────
        flow, flow_per_seed = {}, {}
        if not args.no_power_flow:
            print(f"\n=== power flow: OpenDSS, first {args.pf_n} feeders per arm, source "
                  f"{args.source_pu} pu, same load on every node ===\n")
            real_flow = power_flow({REAL: ctx.floor}, args)      # no sampling: run once
            for g, arms in flow_arms.items():
                flow_per_seed[g] = power_flow(arms, args)
                print(f"  sampling seed {g} solved", flush=True)
            flow = {**flow_summary([real_flow]), **flow_summary(list(flow_per_seed.values()))}
            flow_per_seed = {"real": real_flow, **flow_per_seed}

            print(f"\nmean over sampling seeds {args.gen_seeds}\n")
            print(f"{'kW/node':>8}{'arm':>28}" + "".join(f"{h:>10}" for _, h, _ in FLOW_COLUMNS))
            for kw in args.pf_loads:
                for name in results:
                    row = flow[f"{kw:g} kW | {name}"]
                    print(f"{kw:>8.0f}{name:>28}"
                          + "".join(f"{row[k]['mean']:>10{f}}" for k, _, f in FLOW_COLUMNS))
                print()
            print("a low 'energised' fraction flatters every voltage column: de-energised "
                  "buses carry no load")

        # ── summary on the paper's scale ─────────────────────────────────────
        summary = {}
        for name, r in results.items():
            s = {"realism": r["realism"], "physical": r["physical"]}
            if flow:
                s["operational"] = flow[f"{args.pf_kw:g} kW | {name}"]["operational"]
                s["overall"] = float(np.mean([s["realism"]["mean"], s["physical"]["mean"],
                                              s["operational"]["mean"]]))
            summary[name] = s
        print("\n=== summary, 0-10 (attribute quality on real topology) ===\n")
        print(f"{'set':<26}{'realism':>9}{'sd GIN':>8}{'sd samp.':>9}{'physical':>10}"
              f"{'sd samp.':>9}" + (f"{'operational':>13}{'sd samp.':>9}{'overall':>9}"
                                    if flow else ""))
        for name, s in summary.items():
            line = (f"{name:<26}{s['realism']['mean']:>9.2f}{s['realism']['sd_gin']:>8.2f}"
                    f"{s['realism']['sd_sampling']:>9.2f}{s['physical']['mean']:>10.2f}"
                    f"{s['physical']['sd_sampling']:>9.2f}")
            if flow:
                line += (f"{s['operational']['mean']:>13.2f}"
                         f"{s['operational']['sd_sampling']:>9.2f}{s['overall']:>9.2f}")
            print(line)
        if flow:
            print(f"\noperational at {args.pf_kw:g} kW per node, first {args.pf_n} feeders of "
                  f"every sampling seed")

    with open(os.path.join(args.out, "ladder.json"), "w") as f:
        json.dump({"n": args.n, "seed": args.seed, "gin_seeds": args.gin_seeds,
                   "sampling_seeds": args.gen_seeds, "checkpoints": args.checkpoints,
                   "summary": summary, "results": results, "power_flow": flow,
                   "per_seed": per_seed, "power_flow_per_seed": flow_per_seed}, f, indent=2)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
