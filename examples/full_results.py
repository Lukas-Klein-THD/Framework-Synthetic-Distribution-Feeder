"""
Every results-section number for the two reference models.

    python examples/full_results.py --device cuda --out <dir>
    python examples/full_results.py --device cuda --out <dir> --sections power_flow

Mirrors the sections of ``custom-approach/results/2026-09-30_v2_results`` for
the staged model and the improved FeederGAN (M1+), both trained in this
framework.  Each section writes ``<out>/logs/<section>.log``; all numbers are
also collected in ``<out>/numbers.json``.

    topology            structure metrics, untrained baselines, tree shape (n = 512)
    attributed          attributed metrics, stage 2 on real topology, per-attribute
                        distances and persistence (n = 512)
    electrical          phase subset and ampacity tapering (n = 512)
    phase_constraint    unconstrained / repaired / constrained, all PAIRED (n = 256)
    power_flow          OpenDSS at 10 / 50 / 100 kW per node, 1.03 pu (n = 64)
    baselines           trained vs untrained component per stage, profile echo (n = 256)
    categories          per-category attribute quality (original feeders only)

Sections 1-2 use the same pool split, seeds and sampling order as
``compare.py``, so their numbers are identical to the comparison table.
Power flow uses the custom-approach OpenDSS harness (``--custom-approach``) so
its columns match the earlier power-flow tables exactly.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gridgen import Evaluator                                              # noqa: E402
from gridgen.core.feeder import Feeder                                     # noqa: E402
from plugins.distribution import (                                         # noqa: E402
    AmpacityTapering, AttributeEncoder, PhaseSubset, TensorFeederSource,
    attributed_metrics, structural_metrics)
from plugins.distribution import config as shared                          # noqa: E402
from plugins.distribution.data import SourcePathLabeler                    # noqa: E402
from compare import tree_shape                                             # noqa: E402

AMP_KEYS = ("gin-attributed.prdc.precision", "gin-attributed.prdc.recall",
            "gin-attributed.prdc.coverage", "gin-attributed.prdc.density",
            "gin-attributed.fid.value")
TOPO_KEYS = ("gin.prdc.precision", "gin.prdc.recall", "gin.prdc.coverage",
             "gin.prdc.density", "degree.mmd.value", "spectrum.mmd.value", "gin.fid.value")


# ── helpers ──────────────────────────────────────────────────────────────────
class Context:
    """Data, models and the shared real sets, built once."""

    def __init__(self, args):
        from plugins.feedergan_improved import FeederGANImproved
        from plugins.staged import build_staged
        from plugins.staged.attributes import MDNAttributes

        self.args, self.n, self.seed, self.device = args, args.n, args.seed, args.device
        self.source = TensorFeederSource()
        pool = self.source.sample(2 * self.n, seed=self.seed)
        self.reference, self.floor = pool[:self.n], pool[self.n:]
        self.encoder = AttributeEncoder.fit(self.reference)
        staged_dir = os.path.join(shared.CHECKPOINT_ROOT, "staged")
        topo, attr = (os.path.join(staged_dir, "topology.pt"),
                      os.path.join(staged_dir, "attributes.pt"))
        self.staged = build_staged(topo, attr, device=self.device)
        self.mdn = self.staged.attributes
        self.mdn_constrained = MDNAttributes(checkpoint=attr, device=self.device,
                                             constrain_phase=True)
        self.m1 = FeederGANImproved(os.path.join(shared.CHECKPOINT_ROOT, "feedergan_improved",
                                                 "feedergan_improved_final.pt"),
                                    device=self.device)
        self._samples: dict[str, list[Feeder]] = {}

    def samples(self, name: str) -> list[Feeder]:
        """The n-feeder sets of compare.py (same seeds, same order)."""
        if name not in self._samples:
            gen = {"staged": self.staged, "feedergan-improved": self.m1}[name]
            torch.manual_seed(self.seed)
            self._samples[name] = gen.sample(self.n)
        return self._samples[name]


def evaluate(metrics, reference, feeders, seed, title):
    card = Evaluator(metrics).run(reference, feeders, with_reference=False,
                                  title=title, seed=seed)
    return {k: float(v) for k, v in card.scores.items()}


def random_trees(n, m, attachment, seed):
    """Untrained rooted trees, as custom-approach ``approaches/baselines.py``."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        a = np.zeros((m, m), dtype=np.float32)
        out_degree = np.zeros(m)
        for j in range(1, m):
            if attachment == "uniform":
                parent = rng.integers(0, j)
            else:
                w = out_degree[:j] + 1.0
                parent = rng.choice(j, p=w / w.sum())
            a[parent, j] = 1.0
            out_degree[parent] += 1
        out.append(a)
    return out


def with_attributes(adjacencies, attributes):
    return [Feeder(a, {k: np.asarray(v[k], np.float32) for k in ("length", "ampacity", "phase")})
            for a, v in zip(adjacencies, attributes)]


def generate_attributes(model, adjacencies, seed=None, profile=None, batch=128):
    """Attributes from the staged MDN; ``seed`` makes two calls draw identically."""
    if seed is not None:
        torch.manual_seed(seed)
    out = []
    for i in range(0, len(adjacencies), batch):
        chunk = adjacencies[i:i + batch]
        if profile is None:
            out += model.generate(chunk)
        else:
            A = torch.as_tensor(np.stack(chunk), dtype=torch.float32, device=model.device)
            p = profile[i:i + batch].to(model.device)
            sampled = model.model.sample(A, p, constrain_phase=model.constrain_phase)
            out += [{k: v[j].cpu().numpy() for k, v in sampled.items()} for j in range(A.size(0))]
    return out


def values(feeders, key, skip_root=False):
    return np.concatenate([np.asarray(f.attributes[key], float)[1 if skip_root else 0:]
                           for f in feeders])


def w1(a, b):
    """1-Wasserstein via matched quantiles (custom-approach ``_w1``)."""
    a, b = np.sort(np.ravel(a)), np.sort(np.ravel(b))
    q = (np.arange(2 * max(a.size, b.size)) + 0.5) / (2 * max(a.size, b.size))
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def tv(real, gen, classes):
    pr = np.array([(np.abs(real - c) < 1e-3).mean() for c in classes])
    pg = np.array([(np.abs(gen - c) < 1e-3).mean() for c in classes])
    return float(0.5 * np.abs(pr - pg).sum())


def persistence(feeders, key):
    """Share of parent -> child edges where the child keeps the parent's value."""
    same, total = 0, 0
    for f in feeders:
        v = np.round(np.asarray(f.attributes[key], float), 3)
        parents = np.asarray(f.adjacency).argmax(0)
        same += int((v[1:] == v[parents[1:]]).sum())
        total += len(v) - 1
    return same / max(total, 1)


def validity(feeders):
    out = {}
    for c in (PhaseSubset(), AmpacityTapering()):
        v = c.validate(feeders)
        out[f"{c.name}.edges"], out[f"{c.name}.feeders"] = v.element_rate, v.feeder_rate
    return out


def repair(feeders):
    rule = PhaseSubset()
    return [rule.repair(f) for f in feeders]


# ── sections ─────────────────────────────────────────────────────────────────
def topology(ctx):
    metrics = structural_metrics(seed=ctx.seed)
    m = shared.NUM_NODES
    sets = {
        "real (floor)": ctx.floor,
        "staged": ctx.samples("staged"),
        "feedergan-improved": ctx.samples("feedergan-improved"),
        "uniform random tree": [Feeder(a) for a in random_trees(ctx.n, m, "uniform", ctx.seed)],
        "preferential attachment": [Feeder(a) for a in
                                    random_trees(ctx.n, m, "preferential", ctx.seed)],
    }
    rows = {}
    print(f"structure metrics, reference = {ctx.n} real feeders\n")
    print(f"{'set':<26}" + "".join(f"{k.split('.')[-2] if 'prdc' in k else k.split('.')[0]:>11}"
                                     for k in TOPO_KEYS))
    for name, feeders in sets.items():
        s = evaluate(metrics, ctx.reference, feeders, ctx.seed, name)
        rows[name] = {k: s[k] for k in TOPO_KEYS}
        rows[name].update({f"shape.{k}": v for k, v in tree_shape(feeders).items()})
        print(f"{name:<26}" + "".join(f"{s[k]:>11.4g}" for k in TOPO_KEYS))
    rows["real (reference)"] = {f"shape.{k}": v for k, v in tree_shape(ctx.reference).items()}
    print("\ntree shape (mean per feeder)")
    shape_keys = ["leaf_share", "branching_share", "max_out_degree", "max_depth", "diameter"]
    print(f"{'set':<26}" + "".join(f"{k:>16}" for k in shape_keys))
    for name in ["real (reference)", "real (floor)", "staged", "feedergan-improved",
                 "uniform random tree", "preferential attachment"]:
        print(f"{name:<26}" + "".join(f"{rows[name]['shape.' + k]:>16.4g}" for k in shape_keys))
    return rows


def attributed(ctx):
    metrics = attributed_metrics(ctx.encoder, seed=ctx.seed)
    floor_adj = [np.asarray(f.adjacency) for f in ctx.floor]
    isolated = with_attributes(floor_adj, generate_attributes(ctx.mdn, floor_adj, seed=ctx.seed))
    sets = {
        "real (floor)": ctx.floor,
        "staged (full)": ctx.samples("staged"),
        "staged stage 2 on real topology": isolated,
        "feedergan-improved": ctx.samples("feedergan-improved"),
    }
    amps = ctx.encoder.ampacities
    ref = ctx.reference
    rows = {}
    print(f"attributed metrics, reference = {ctx.n} real feeders\n")
    print(f"{'set':<34}{'precision':>10}{'recall':>10}{'coverage':>10}{'density':>10}{'FID':>10}")
    for name, feeders in sets.items():
        s = evaluate(metrics, ref, feeders, ctx.seed, name)
        rows[name] = {k: s[k] for k in AMP_KEYS}
        print(f"{name:<34}" + "".join(f"{s[k]:>10.4g}" for k in AMP_KEYS))

    print("\nper-attribute distance to the reference (all nodes) and persistence "
          "(share of edges where the child keeps the parent's value)\n")
    print(f"{'set':<34}{'length W1 km':>13}{'amp TV':>9}{'phase TV':>10}"
          f"{'amp persist':>12}{'phase persist':>14}")
    for name, feeders in {"real (reference)": ref, **sets}.items():
        d = {
            "length_w1_km": w1(values(ref, "length"), values(feeders, "length")),
            "ampacity_tv": tv(values(ref, "ampacity"), values(feeders, "ampacity"), amps),
            "phase_tv": tv(values(ref, "phase"), values(feeders, "phase"), range(7)),
            "ampacity_persistence": persistence(feeders, "ampacity"),
            "phase_persistence": persistence(feeders, "phase"),
        }
        rows.setdefault(name, {}).update(d)
        print(f"{name:<34}{d['length_w1_km']:>13.4f}{d['ampacity_tv']:>9.3f}"
              f"{d['phase_tv']:>10.3f}{d['ampacity_persistence']:>12.3f}"
              f"{d['phase_persistence']:>14.3f}")
    ctx._isolated = isolated
    return rows


def electrical(ctx):
    floor_adj = [np.asarray(f.adjacency) for f in ctx.floor]
    isolated = getattr(ctx, "_isolated", None) or with_attributes(
        floor_adj, generate_attributes(ctx.mdn, floor_adj, seed=ctx.seed))
    sets = {"real (reference)": ctx.reference,
            "staged, real topology": isolated,
            "staged (full)": ctx.samples("staged"),
            "feedergan-improved": ctx.samples("feedergan-improved")}
    rows = {}
    print(f"{'set':<26}{'phase edges':>13}{'phase feeders':>15}"
          f"{'taper edges':>13}{'taper feeders':>15}")
    for name, feeders in sets.items():
        v = validity(feeders)
        rows[name] = v
        print(f"{name:<26}{v['phase-subset.edges']:>13.4f}{v['phase-subset.feeders']:>15.4f}"
              f"{v['ampacity-tapering.edges']:>13.4f}{v['ampacity-tapering.feeders']:>15.4f}")
    # where the phase violations sit
    for name in ("staged (full)", "feedergan-improved"):
        rule, at_root, total = PhaseSubset(), 0, 0
        for f in sets[name]:
            bad = ~rule.check(f)
            parents = np.asarray(f.adjacency).argmax(0)[1:]
            at_root += int((bad & (parents == 0)).sum())
            total += int(bad.sum())
        rows[name]["phase_violations_at_root_children"] = at_root / max(total, 1)
        print(f"{name}: {total} violating edges, {at_root} of them below the root")
    return rows


def phase_constraint(ctx, n=256):
    n = min(n, ctx.n)
    """All arms paired: same topology; constrained and unconstrained share every draw."""
    reference = ctx.reference[:n]
    metrics = attributed_metrics(ctx.encoder, seed=ctx.seed)
    real_phase = values(reference, "phase")
    rows = {}
    print(f"{'seed':<6}{'arm':<34}{'edge ok':>9}{'feeder ok':>11}{'phase TV':>10}"
          f"{'attr FID':>10}{'precision':>11}{'recall':>9}")
    for seed in (0, 1):
        torch.manual_seed(seed)
        adj = ctx.staged.topology.sample_topology(n)
        free = with_attributes(adj, generate_attributes(ctx.mdn, adj, seed=1000 + seed))
        constrained = with_attributes(adj, generate_attributes(ctx.mdn_constrained, adj,
                                                               seed=1000 + seed))
        torch.manual_seed(seed)
        m1 = ctx.m1.sample(n)
        arms = {"staged, unconstrained": free,
                "staged, repaired post hoc": repair(free),
                "staged, constrained sampling": constrained,
                "feedergan-improved, unconstrained": m1,
                "feedergan-improved, repaired post hoc": repair(m1)}
        for name, feeders in arms.items():
            s = evaluate(metrics, reference, feeders, ctx.seed, name)
            v = validity(feeders)
            row = {"edge_ok": v["phase-subset.edges"], "feeder_ok": v["phase-subset.feeders"],
                   "phase_tv": tv(real_phase, values(feeders, "phase"), range(7)),
                   "fid": s["gin-attributed.fid.value"],
                   "precision": s["gin-attributed.prdc.precision"],
                   "recall": s["gin-attributed.prdc.recall"]}
            rows[f"seed {seed} | {name}"] = row
            print(f"{seed:<6}{name:<34}{row['edge_ok']:>9.4f}{row['feeder_ok']:>11.4f}"
                  f"{row['phase_tv']:>10.4f}{row['fid']:>10.1f}{row['precision']:>11.4f}"
                  f"{row['recall']:>9.4f}")
        # how many phase values differ between the paired arms
        diff = np.mean([(np.asarray(a.attributes["phase"]) != np.asarray(b.attributes["phase"])).mean()
                        for a, b in zip(free, constrained)])
        rows[f"seed {seed} | changed phase share (constrained vs unconstrained)"] = float(diff)
        print(f"      constrained vs unconstrained: {diff:.4f} of phase values differ")
    return rows


def power_flow(ctx, n=64, loads=(10.0, 50.0, 100.0), source_pu=1.03):
    sys.path.append(os.path.abspath(ctx.args.custom_approach))
    from postprocessing.powerflow import LinecodeLibrary, run_feeder

    lib = LinecodeLibrary(ctx.args.linecodes)
    staged, m1 = ctx.samples("staged")[:n], ctx.samples("feedergan-improved")[:n]
    arms = {"real": ctx.floor[:n], "staged": staged, "staged + phase repair": repair(staged),
            "feedergan-improved": m1, "feedergan-improved + phase repair": repair(m1)}

    def matrix(f):
        return np.stack([f.attributes["length"], f.attributes["ampacity"],
                         f.attributes["phase"]], axis=1).astype(np.float32)

    rows = {}
    print(f"OpenDSS, {n} feeders per arm, source {source_pu} pu, same load on every node\n")
    print(f"{'kW/node':>8}{'arm':>36}{'energised':>10}{'Vmin':>8}{'ANSI-A':>8}{'ANSI-B':>8}"
          f"{'unbal%':>8}{'unb<=3%':>9}{'loading':>9}{'overload':>10}")
    for kw in loads:
        for name, feeders in arms.items():
            res = [run_feeder(np.asarray(f.adjacency), matrix(f), lib, kw_per_node=kw,
                              length_units="km", source_pu=source_pu) for f in feeders]

            def m(key):
                return float(np.nanmean([r.get(key, np.nan) for r in res]))
            row = {"converged": float(np.mean([r["converged"] for r in res])),
                   "energised": m("energised_frac"), "vmin": m("vmin"),
                   "ansi_a": m("in_band_frac"), "ansi_b": m("rangeB_frac"),
                   "unbalance_pct": m("unbalance_mean"), "unbalance_ok": m("unbalance_ok_frac"),
                   "loading": m("loading_mean"), "overload": m("overload_frac")}
            rows[f"{kw:g} kW | {name}"] = row
            print(f"{kw:>8.0f}{name:>36}{row['energised']:>10.3f}{row['vmin']:>8.4f}"
                  f"{row['ansi_a']:>8.3f}{row['ansi_b']:>8.3f}{row['unbalance_pct']:>8.2f}"
                  f"{row['unbalance_ok']:>9.3f}{row['loading']:>9.3f}{row['overload']:>10.3f}")
        print()
    print("a low 'energised' fraction flatters every voltage column: de-energised buses "
          "carry no load")
    return rows


def _echo(ctx, adjacencies, real, seed):
    """Profile echo: i.i.d. attributes matching a drawn profile and the real marginals."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    prof = ctx.mdn.profile
    raw = (prof.sample(len(adjacencies)).cpu() * prof.std.cpu() + prof.mean.cpu()).numpy()
    amp_values, amp_counts = np.unique(np.round(values(real, "ampacity"), 3), return_counts=True)
    phase_counts = np.bincount(values(real, "phase").astype(int), minlength=7).astype(float)
    out = []
    for a, r in zip(adjacencies, raw):
        m = a.shape[0]
        out.append({"length": np.clip(10.0 ** rng.normal(r[0], max(r[1], 1e-3), m), 0.0, None),
                    "ampacity": rng.choice(amp_values, m, p=amp_counts / amp_counts.sum()),
                    "phase": rng.choice(7, m, p=phase_counts / phase_counts.sum()).astype(float)})
    return out


def baselines(ctx, n=256):
    n = min(n, ctx.n)
    reference, floor = ctx.reference[:n], ctx.floor[:n]
    metrics = attributed_metrics(ctx.encoder, seed=ctx.seed)
    torch.manual_seed(ctx.seed)
    gan = ctx.staged.topology.sample_topology(n)
    rnd = random_trees(n, shared.NUM_NODES, "uniform", ctx.seed)
    sets = {
        "real (floor)": floor,
        "GAN + MDN (staged)": with_attributes(gan, generate_attributes(ctx.mdn, gan, seed=ctx.seed)),
        "GAN + i.i.d. echo": with_attributes(gan, _echo(ctx, gan, reference, ctx.seed)),
        "random tree + MDN": with_attributes(rnd, generate_attributes(ctx.mdn, rnd, seed=ctx.seed)),
        "random tree + echo": with_attributes(rnd, _echo(ctx, rnd, reference, ctx.seed)),
        "feedergan-improved": ctx.samples("feedergan-improved")[:n],
    }
    rows = {}
    print(f"{'set':<26}{'precision':>10}{'recall':>10}{'coverage':>10}{'FID':>10}"
          f"{'amp persist':>12}{'phase feeders':>14}")
    for name, feeders in sets.items():
        s = evaluate(metrics, reference, feeders, ctx.seed, name)
        row = {k: s[k] for k in AMP_KEYS}
        row["ampacity_persistence"] = persistence(feeders, "ampacity")
        row["phase_feeder_validity"] = validity(feeders)["phase-subset.feeders"]
        rows[name] = row
        print(f"{name:<26}{s[AMP_KEYS[0]]:>10.4f}{s[AMP_KEYS[1]]:>10.4f}{s[AMP_KEYS[2]]:>10.4f}"
              f"{s[AMP_KEYS[4]]:>10.1f}{row['ampacity_persistence']:>12.3f}"
              f"{row['phase_feeder_validity']:>14.3f}")
    rows["real (reference)"] = {"ampacity_persistence": persistence(reference, "ampacity"),
                                "phase_feeder_validity": validity(reference)["phase-subset.feeders"]}
    print(f"{'real (reference)':<26}{'':>40}{rows['real (reference)']['ampacity_persistence']:>12.3f}"
          f"{rows['real (reference)']['phase_feeder_validity']:>14.3f}")
    return rows


def categories(ctx, per_category=114):
    """Attribute quality per category on original feeders, as evaluate_by_category.py."""
    from plugins.staged.profile import FeederProfile, compute_raw

    originals = TensorFeederSource(include_decomposed=False, labeler=SourcePathLabeler())
    feeders = [originals[i] for i in range(len(originals))]
    by_cat: dict[str, list[Feeder]] = {}
    for f in feeders:
        by_cat.setdefault(f.category, []).append(f)
    print({c: len(v) for c, v in sorted(by_cat.items(), key=lambda kv: str(kv[0]))})

    # the trained profile pool carries no labels; rebuild it from the labelled
    # originals with the TRAINING standardisation, which leaves the model's
    # conditioning space unchanged
    base = ctx.mdn.profile
    raw = np.stack([compute_raw(f.attributes) for f in feeders])
    mean, std = base.mean.cpu(), base.std.cpu()
    pool = (torch.from_numpy(raw).float() - mean) / std
    labelled = FeederProfile(mean, std, pool, [f.category for f in feeders])

    def summary(real, gen):
        r_len, g_len = values(real, "length"), values(gen, "length")
        amps = np.unique(np.round(values(real, "ampacity"), 3))
        return {"length_w1_km": w1(r_len, g_len),
                "ampacity_tv": tv(np.round(values(real, "ampacity"), 3),
                                  np.round(values(gen, "ampacity"), 3), amps),
                "phase_tv": tv(values(real, "phase"), values(gen, "phase"), range(7)),
                "three_phase_share": float((np.round(values(gen, "phase")) == 6).mean()),
                "mean_length_km": float(g_len.mean())}

    rows = {}
    print(f"\n{'category':<16}{'n':>5}  {'arm':<34}{'length W1':>10}{'amp TV':>8}"
          f"{'phase TV':>9}{'3ph share':>10}{'mean len':>9}")
    for cat in ("industrial", "rural", "urban-suburban"):
        pool_cat = by_cat.get(cat, [])
        if not pool_cat:
            continue
        g = torch.Generator().manual_seed(ctx.seed)
        k = min(per_category, len(pool_cat))
        real = [pool_cat[i] for i in torch.randperm(len(pool_cat), generator=g)[:k].tolist()]
        adj = [np.asarray(f.adjacency) for f in real]
        true_profile = labelled.encode(real).float()             # teacher forcing
        torch.manual_seed(ctx.seed)
        strat = labelled.sample(k, category=cat)
        torch.manual_seed(ctx.seed)
        gen_adj = ctx.staged.topology.sample_topology(k)
        torch.manual_seed(ctx.seed)
        boot = base.sample(k)
        arms = {
            "staged, bootstrapped profile": with_attributes(
                adj, generate_attributes(ctx.mdn, adj, seed=ctx.seed, profile=boot)),
            "staged, stratified profile": with_attributes(
                adj, generate_attributes(ctx.mdn, adj, seed=ctx.seed, profile=strat)),
            "staged, stratified, generated topology": with_attributes(
                gen_adj, generate_attributes(ctx.mdn, gen_adj, seed=ctx.seed, profile=strat)),
            "staged, true profile": with_attributes(
                adj, generate_attributes(ctx.mdn, adj, seed=ctx.seed, profile=true_profile)),
            "feedergan-improved (unconditional)": ctx.samples("feedergan-improved")[:k],
        }
        for name, gen in arms.items():
            s = summary(real, gen)
            rows[f"{cat} | {name}"] = s
            print(f"{cat:<16}{k:>5}  {name:<34}{s['length_w1_km']:>10.3f}{s['ampacity_tv']:>8.3f}"
                  f"{s['phase_tv']:>9.3f}{s['three_phase_share']:>10.3f}{s['mean_length_km']:>9.4f}")
        r_len = values(real, "length")
        rows[f"{cat} | real"] = {"three_phase_share": float((np.round(values(real, "phase")) == 6).mean()),
                                 "mean_length_km": float(r_len.mean())}
        print(f"{'':<16}{'':>5}  {'REAL ' + cat:<34}{'':>27}"
              f"{rows[f'{cat} | real']['three_phase_share']:>10.3f}"
              f"{rows[f'{cat} | real']['mean_length_km']:>9.4f}\n")
    print("distances are to that category's own real feeders (lower is better)")
    return rows


SECTIONS = {"topology": topology, "attributed": attributed, "electrical": electrical,
            "phase_constraint": phase_constraint, "power_flow": power_flow,
            "baselines": baselines, "categories": categories}


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--sections", nargs="+", default=list(SECTIONS), choices=list(SECTIONS))
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--custom-approach", default=os.path.join(REPO, "..", "custom-approach"))
    p.add_argument("--linecodes", default=os.path.join(
        REPO, "..", "paper-implementation", "data", "dss_data", "GSO_Base_Network",
        "Urban-Suburban", "Linecodes.dss"))
    args = p.parse_args()

    logs = os.path.join(args.out, "logs")
    os.makedirs(logs, exist_ok=True)
    numbers_path = os.path.join(args.out, "numbers.json")
    numbers = json.load(open(numbers_path)) if os.path.isfile(numbers_path) else {}

    ctx = Context(args)
    for name in args.sections:
        with open(os.path.join(logs, f"{name}.log"), "w", encoding="utf-8") as log, \
                contextlib.redirect_stdout(_Tee(sys.__stdout__, log)):
            print(f"=== {name} ===")
            numbers[name] = SECTIONS[name](ctx)
        with open(numbers_path, "w") as f:
            json.dump(numbers, f, indent=2, default=float)
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
