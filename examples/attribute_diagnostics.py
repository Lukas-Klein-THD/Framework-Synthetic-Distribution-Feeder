"""
Which attributes and dependencies drive the staged model's low attributed precision?

    python examples/attribute_diagnostics.py --device cuda --out <results dir>

Diagnostics only: the models are used as trained, nothing about them changes.
Every experiment runs on **fixed held-out reference topologies** (the comparison
half of the split), so topology errors are excluded by construction, and every
variant is scored against the same reference half, with the same attribute
encoding (fitted on training data) and the same GIN initialisations, so the
evaluation geometry (embedding and reference neighbourhoods) is identical for all.
Generated attributes are drawn with several sampling seeds; reported spreads are
over sampling seeds (each value first averaged over the GIN seeds).

Experiment 1, direct attribute replacement (an intervention).
    Real feeders whose attributes in a subset S are replaced, node by node on the
    same topology, by the values the staged model generates for that topology.
    Precision drop dP_S = P(real) - P(S), interaction I_AB = dP_AB - dP_A - dP_B,
    and the same for the FD increase.  Mixed feeders are interventions, not
    generator samples, and precision saturates once a sample leaves the reference
    manifold, so interaction terms describe the metric's response, not by
    themselves a learned dependency.  Inserting real values for S into generated
    feeders equals replacing the complement of S, so the reverse direction is
    read off the same sets.

Experiment 2, conditional regeneration (diagnostic).
    Sampled vs true feeder profile, own vs real parent values.  With real parent
    values most nodes copy a real value, so these variants leak real information:
    they diagnose the per-node conditional, they are not generative performance.

Experiment 3, dependency statistics.
    Ampacity-phase dependence (normalised mutual information, overall, within depth
    bands and within categories), ampacity by phase count and depth, three-phase
    share by depth, changes at branches vs along runs, persistence, tapering,
    length-ampacity rank correlation; 95 % bootstrap intervals over feeders, a
    second real sample as the noise level.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gridgen.core.feeder import Feeder                                     # noqa: E402
from plugins.distribution import attributed_metrics                        # noqa: E402
from plugins.distribution.data import SourcePathLabeler                    # noqa: E402
from plugins.distribution.metrics import realism_score                     # noqa: E402
from plugins.staged.layers import parent_index                             # noqa: E402
from plugins.staged.profile import compute_raw                             # noqa: E402
from full_results import Context, _Tee, evaluate, generate_attributes      # noqa: E402

ATTRS = ("length", "ampacity", "phase")
SHORT = {"length": "L", "ampacity": "A", "phase": "P"}
KEYS = ("gin-attributed.prdc.precision", "gin-attributed.prdc.recall",
        "gin-attributed.fid.value")
PHASE_COUNT = np.array([1, 1, 1, 2, 2, 2, 3])
DEPTH_BINS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 30), (30, 10 ** 6)]
CATEGORIES = ("urban-suburban", "rural")          # industrial: too few held-out feeders
VARIANTS = ("sampled profile", "true profile", "sampled profile, real parents",
            "true profile, real parents")


# ── feeders ──────────────────────────────────────────────────────────────────
def attrs_of(feeder):
    return {k: np.asarray(feeder.attributes[k], np.float32) for k in ATTRS}


def build(adjacencies, attributes):
    return [Feeder(np.asarray(a), {k: np.asarray(d[k], np.float32) for k in ATTRS})
            for a, d in zip(adjacencies, attributes)]


def mix(base, donor, replace):
    """Per node and per feeder: attributes in ``replace`` from donor, others from base."""
    return [{k: (d[k] if k in replace else b[k]) for k in ATTRS} for b, d in zip(base, donor)]


@torch.no_grad()
def sample_with_real_parents(mdn, adjacencies, real_attrs, profile, seed, batch=128):
    """Ampacity and phase drawn per node given the *real* parent value (one-step
    conditional); length as usual.  Uses the trained modules unchanged."""
    torch.manual_seed(seed)
    model, dev = mdn.model, mdn.device
    out = []
    for i in range(0, len(adjacencies), batch):
        A = torch.as_tensor(np.stack(adjacencies[i:i + batch]), dtype=torch.float32, device=dev)
        b, m = A.shape[:2]
        h = model._context(A, profile[i:i + b].to(dev))
        parents = parent_index(A)
        sampled = {"length": model.heads["length"].sample(h)}
        for name in model.MARKOV:
            head = model.heads[name]
            real = torch.as_tensor(np.stack([r[name] for r in real_attrs[i:i + b]]),
                                   dtype=torch.float32, device=dev)
            parent_cls = torch.gather(head.to_index(real), 1, parents).clone()
            parent_cls[:, 0] = head.n_classes                       # root sentinel
            x = torch.cat([h, model.parent_embed[name](parent_cls)], dim=-1)
            gate_logit, class_logits = head.forward_parts(x)
            change = torch.bernoulli(torch.sigmoid(gate_logit)).bool()
            change[:, 0] = True                                     # the root always draws
            drawn = torch.multinomial(class_logits.softmax(-1).reshape(-1, head.n_classes),
                                      1).view(b, m)
            cls = torch.where(change, drawn, parent_cls.clamp_max(head.n_classes - 1))
            sampled[name] = head.values[cls]
        out += [{k: v[j].cpu().numpy() for k, v in sampled.items()} for j in range(b)]
    return out


def generate_variants(ctx, adj, real_attrs, true_profile, seed):
    gen = {"sampled profile": generate_attributes(ctx.mdn, adj, seed=seed),
           "true profile": generate_attributes(ctx.mdn, adj, seed=seed, profile=true_profile)}
    torch.manual_seed(seed)
    sampled_profile = ctx.mdn.profile.sample(len(adj))
    gen["sampled profile, real parents"] = sample_with_real_parents(
        ctx.mdn, adj, real_attrs, sampled_profile, seed)
    gen["true profile, real parents"] = sample_with_real_parents(
        ctx.mdn, adj, real_attrs, true_profile, seed)
    return gen


# ── aggregation ──────────────────────────────────────────────────────────────
def stat(values):
    values = [float(v) for v in values]
    return {"values": values, "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def over_seeds(matrix):
    """[generation seed x GIN seed] -> mean, std over generation seeds of the GIN
    means, and the mean std over GIN seeds."""
    m = np.asarray(matrix, float)
    per_gen = m.mean(axis=1)
    return {"mean": float(m.mean()),
            "std_sampling": float(per_gen.std(ddof=1)) if len(per_gen) > 1 else 0.0,
            "std_embedding": float(m.std(axis=1, ddof=1).mean()) if m.shape[1] > 1 else 0.0,
            "per_sampling_seed": per_gen.tolist()}


# ── dependency statistics ────────────────────────────────────────────────────
def _ranks(x):
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + ends - 1) / 2.0)[inverse]


def _spearman(x, y):
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def _nmi(a, b):
    if len(a) == 0:
        return float("nan")
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    joint = np.zeros((ia.max() + 1, ib.max() + 1))
    np.add.at(joint, (ia, ib), 1)
    p = joint / joint.sum()
    pa, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = float((p[nz] * np.log(p[nz] / (pa @ pb)[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / np.sqrt(ha * hb) if ha > 0 and hb > 0 else 0.0


def node_arrays(feeders, categories, parent_attrs=None):
    """Per-feeder arrays over non-root nodes, computed once (bootstrap resamples them).

    ``parent_attrs`` gives the parent values a node was conditioned on when they
    differ from the feeder's own (real-parent variants).
    """
    out = []
    for i, f in enumerate(feeders):
        a = np.asarray(f.adjacency) != 0
        parents = a.argmax(0)
        outdeg = a.sum(1)
        d = np.zeros(len(parents), int)
        for j in range(1, len(parents)):
            d[j] = d[parents[j]] + 1
        amp = np.round(np.asarray(f.attributes["ampacity"], float), 1)
        ph = np.asarray(f.attributes["phase"]).astype(int)
        src = parent_attrs[i] if parent_attrs is not None else f.attributes
        amp_par = np.round(np.asarray(src["ampacity"], float), 1)
        ph_par = np.asarray(src["phase"]).astype(int)
        n = np.arange(1, len(parents))
        out.append({"L": np.asarray(f.attributes["length"], float)[n], "A": amp[n],
                    "P": ph[n], "depth": d[n], "branch": outdeg[parents[n]] >= 2,
                    "same_A": amp[n] == amp_par[parents[n]],
                    "same_P": ph[n] == ph_par[parents[n]],
                    "taper": amp[n] <= amp_par[parents[n]] + 1e-6,
                    "category": categories[i]})
    return out


def stats_from(arrays):
    cat = lambda k: np.concatenate([x[k] for x in arrays])            # noqa: E731
    L, A, P, depth = cat("L"), cat("A"), cat("P"), cat("depth")
    branch, same_A, same_P, taper = cat("branch"), cat("same_A"), cat("same_P"), cat("taper")
    category = np.concatenate([np.full(len(x["A"]), x["category"] or "") for x in arrays])
    count = PHASE_COUNT[np.clip(P, 0, 6)]
    log_a = np.log10(np.clip(A, 1.0, None))
    bins = [(depth >= lo) & (depth < hi) for lo, hi in DEPTH_BINS]

    def mean_or_nan(x, sel):
        return float(x[sel].mean()) if sel.any() else float("nan")

    within_depth = [(_nmi(A[s], P[s]), s.sum()) for s in bins if s.sum() > 20]
    return {
        "nmi_ampacity_phase": _nmi(A, P),
        "nmi_ampacity_phase_within_depth": float(
            sum(v * w for v, w in within_depth) / sum(w for _, w in within_depth)),
        **{f"nmi_ampacity_phase_{c}": _nmi(A[category == c], P[category == c])
           for c in CATEGORIES},
        "log_ampacity_by_phase_count": [mean_or_nan(log_a, count == c) for c in (1, 2, 3)],
        **{f"log_ampacity_{c}_phase_by_depth": [mean_or_nan(log_a, s & (count == k))
                                                for s in bins]
           for c, k in (("single", 1), ("three", 3))},
        "three_phase_share_by_depth": [mean_or_nan((count == 3).astype(float), s) for s in bins],
        "log_ampacity_by_depth": [mean_or_nan(log_a, s) for s in bins],
        "ampacity_change_at_branches": float(1 - same_A[branch].mean()),
        "ampacity_change_along_runs": float(1 - same_A[~branch].mean()),
        "phase_change_at_branches": float(1 - same_P[branch].mean()),
        "phase_change_along_runs": float(1 - same_P[~branch].mean()),
        "ampacity_persistence": float(same_A.mean()),
        "phase_persistence": float(same_P.mean()),
        "tapering_edges": float(taper.mean()),
        "spearman_length_ampacity": _spearman(L, A),
    }


def bootstrap(arrays, n_boot=200, seed=0):
    """95 % percentile intervals over feeders for every statistic."""
    rng = np.random.default_rng(seed)
    draws = [stats_from([arrays[i] for i in rng.integers(0, len(arrays), len(arrays))])
             for _ in range(n_boot)]
    ci = {}
    for k, v in draws[0].items():
        x = np.array([d[k] for d in draws], float)
        lo, hi = np.nanpercentile(x, 2.5, axis=0), np.nanpercentile(x, 97.5, axis=0)
        ci[k] = (lo.tolist(), hi.tolist()) if x.ndim > 1 else (float(lo), float(hi))
    return ci


def fmt(x):
    if isinstance(x, list):
        return "[" + ", ".join("  nan" if not np.isfinite(y) else f"{y:.2f}" for y in x) + "]"
    return f"{x:.3f}"


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="GIN embedding seeds")
    p.add_argument("--gen-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="sampling seeds of the generated attributes")
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--seed", type=int, default=0, help="sampling seed of the real sets")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--device", default="cpu")
    p.add_argument("--all-data", action="store_true")
    args = p.parse_args()
    args.custom_approach, args.linecodes = None, None
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)

    with open(os.path.join(args.out, "logs", "attribute_diagnostics.log"), "w",
              encoding="utf-8") as log, contextlib.redirect_stdout(_Tee(sys.__stdout__, log)):
        ctx = Context(args)
        real = ctx.floor                                   # fixed held-out topologies
        adj = [np.asarray(f.adjacency) for f in real]
        real_attrs = [attrs_of(f) for f in real]
        base = ctx.mdn.profile
        raw_profile = torch.from_numpy(np.stack([compute_raw(f.attributes) for f in real])).float()
        true_profile = (raw_profile - base.mean.cpu()) / base.std.cpu()
        labeler = SourcePathLabeler()
        cat_real = [labeler.label(f) for f in real]
        cat_ref = [labeler.label(f) for f in ctx.reference]

        subsets = [s for r in range(4) for s in itertools.combinations(ATTRS, r)]
        label = {s: "+".join(SHORT[a] for a in s) or "none" for s in subsets}
        set_names = [f"replace {label[s]}" for s in subsets] + \
                    [f"regenerate: {v}" for v in VARIANTS]
        print(f"{len(set_names)} sets of {len(real)} feeders on fixed held-out topologies; "
              f"sampling seeds {args.gen_seeds}, embedding seeds {args.seeds}\n")

        # scores[name][gen seed index][gin seed index] = (P, R, FD)
        scores = {name: [] for name in set_names}
        gens = {}
        for gi, g in enumerate(args.gen_seeds):
            gen = generate_variants(ctx, adj, real_attrs, true_profile, g)
            gens[g] = gen
            sets = {f"replace {label[s]}": build(adj, mix(real_attrs, gen["sampled profile"], s))
                    for s in subsets}
            sets.update({f"regenerate: {v}": build(adj, gen[v]) for v in VARIANTS})
            for name in set_names:
                if name == "replace none" and gi > 0:          # real: independent of g
                    scores[name].append(scores[name][0])
                    continue
                row = []
                for e in args.seeds:
                    s = evaluate(attributed_metrics(ctx.encoder, seed=e), ctx.reference,
                                 sets[name], ctx.seed, name)
                    row.append(tuple(s[k] for k in KEYS))
                scores[name].append(row)
            print(f"  sampling seed {g} scored", flush=True)

        arr = {n: np.array(v, float) for n, v in scores.items()}     # [g, e, 3]
        P_real, R_real, FD_real = (arr["replace none"][..., i] for i in range(3))

        def score_matrix(name):
            a = arr[name]
            return np.vectorize(realism_score)(a[..., 0], a[..., 1], P_real, R_real)

        results = {n: {"precision": over_seeds(arr[n][..., 0]), "recall": over_seeds(arr[n][..., 1]),
                       "fd": over_seeds(arr[n][..., 2]), "score": over_seeds(score_matrix(n))}
                   for n in set_names}

        def dP(s):
            return P_real - arr[f"replace {label[s]}"][..., 0]

        def dFD(s):
            return arr[f"replace {label[s]}"][..., 2] - FD_real

        singles = [(a,) for a in ATTRS]
        drops = {label[s]: over_seeds(dP(s)) for s in subsets if s}
        fd_increase = {label[s]: over_seeds(dFD(s)) for s in subsets if s}
        inter, inter_fd = {}, {}
        for a, b in itertools.combinations(ATTRS, 2):
            k = f"{SHORT[a]},{SHORT[b]}"
            inter[k] = over_seeds(dP((a, b)) - dP((a,)) - dP((b,)))
            inter_fd[k] = over_seeds(dFD((a, b)) - dFD((a,)) - dFD((b,)))
        inter["L,A,P (vs singles)"] = over_seeds(dP(ATTRS) - sum(dP(s) for s in singles))
        inter_fd["L,A,P (vs singles)"] = over_seeds(dFD(ATTRS) - sum(dFD(s) for s in singles))
        P_gen = arr["replace L+A+P"][..., 0]

        def complement(s):
            return tuple(a for a in ATTRS if a not in s)
        recovery = {label[s]: over_seeds(arr[f"replace {label[complement(s)]}"][..., 0] - P_gen)
                    for s in subsets if s}

        def show(v, d=3):
            return f"{v['mean']:+.{d}f} ± {v['std_sampling']:.{d}f}"

        def row(name):
            r = results[name]
            return (f"{name:<38}{r['precision']['mean']:>7.3f}±{r['precision']['std_sampling']:.3f}"
                    f"{r['recall']['mean']:>8.3f}±{r['recall']['std_sampling']:.3f}"
                    f"{r['fd']['mean']:>8.0f}±{r['fd']['std_sampling']:<5.0f}"
                    f"{r['score']['mean']:>6.2f}±{r['score']['std_sampling']:.2f}")
        head = f"{'set':<38}{'precision':>13}{'recall':>14}{'FD':>14}{'score':>11}"
        print("\n± = std over sampling seeds (each value averaged over the embedding seeds)")
        print("\n=== Experiment 1: direct replacement (real feeders, attributes S from the model) ===")
        print(head)
        for s in subsets:
            print(row(f"replace {label[s]}"))
        print("\nprecision drop dP_S = P(real) - P(S)        FD increase dFD_S = FD(S) - FD(real)")
        for k in drops:
            print(f"  {k:<8} dP {show(drops[k])}      dFD {show(fd_increase[k], 0)}")
        print("interaction I = d_AB - d_A - d_B")
        for k in inter:
            print(f"  {k:<20} I_P {show(inter[k])}      I_FD {show(inter_fd[k], 0)}")
        print("\n=== Experiment 1b: reverse (generated feeders, real values inserted for S) ===")
        print("identical to replacing the complement of S in real feeders (rows above)")
        for k, v in recovery.items():
            print(f"  +P_{k:<8} {show(v)}")
        print("\n=== Experiment 2: conditional regeneration on fixed topology (diagnostic) ===")
        print(head)
        for v in VARIANTS:
            print(row(f"regenerate: {v}"))

        print("\n=== Experiment 3: dependency statistics (95 % bootstrap over feeders) ===")
        g0 = args.gen_seeds[0]
        dep_arrays = {"reference": node_arrays(ctx.reference, cat_ref),
                      "real (comparison)": node_arrays(real, cat_real)}
        for v in VARIANTS:
            dep_arrays[f"generated: {v}"] = node_arrays(
                build(adj, gens[g0][v]), cat_real, real_attrs if "real parents" in v else None)
        deps = {k: stats_from(v) for k, v in dep_arrays.items()}
        cis = {k: bootstrap(v, args.n_boot) for k, v in dep_arrays.items()}
        # spread over sampling seeds for the generated variants
        seed_spread = {}
        for v in VARIANTS:
            per = [stats_from(node_arrays(build(adj, gens[g][v]), cat_real,
                                          real_attrs if "real parents" in v else None))
                   for g in args.gen_seeds]
            seed_spread[f"generated: {v}"] = {
                k: (np.nanstd(np.array([x[k] for x in per], float), axis=0, ddof=1).tolist()
                    if isinstance(per[0][k], list)
                    else float(np.std([x[k] for x in per], ddof=1)))
                for k in per[0]}
        for k, v in deps.items():
            print(f"\n{k}  (sampling seed {g0} for generated sets)")
            for s, x in v.items():
                lo, hi = cis[k][s]
                print(f"  {s:<38} {fmt(x)}   95% CI {fmt(lo)} .. {fmt(hi)}")

    with open(os.path.join(args.out, "diagnostics.json"), "w") as f:
        json.dump({"embedding_seeds": args.seeds, "sampling_seeds": args.gen_seeds,
                   "n": args.n, "n_boot": args.n_boot, "sets": results,
                   "precision_drop": drops, "fd_increase": fd_increase,
                   "interaction_precision": inter, "interaction_fd": inter_fd,
                   "precision_recovery": recovery, "dependencies": deps,
                   "dependency_ci95": cis, "dependency_std_over_sampling_seeds": seed_spread},
                  f, indent=2)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
