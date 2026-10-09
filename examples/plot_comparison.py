"""
Figures for the model comparison, in the IEEE one-column style.

    python examples/plot_comparison.py --device cuda

Reads ``scores.json`` written by ``compare.py`` and draws the same sample sets
again (same seed, same sampling order, so the feeders are the ones that were
scored).  Writes to ``<results>/figures/``:

    fidelity_diversity   precision against recall, structure and attributed
    distributions        out-degree, line length, ampacity and phase per model
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style                                                         # noqa: E402
from plugins.distribution import TensorFeederSource                        # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Display name and colour per scored set; the floor stands in for real data.
STYLE = {
    "real (floor)": ("Real feeders", "0.45", "o"),
    "staged": ("Staged model", "#1f5fa8", "s"),
    "feedergan-improved": ("Improved FeederGAN", "#d1731d", "^"),
}


def fidelity_diversity(results, out_dir):
    fig, axes = paper_style.subplots(1, 2, 1.75, sharex=True, sharey=True)
    for ax, (prefix, title) in zip(axes, [("gin", "Structure"),
                                          ("gin-attributed", "Structure and attributes")]):
        for name, (label, colour, marker) in STYLE.items():
            r = results[name]
            ax.scatter(r[f"{prefix}.prdc.precision"], r[f"{prefix}.prdc.recall"],
                       color=colour, marker=marker, s=18, label=label, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Precision")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.grid(True, alpha=0.4)
    axes[0].set_ylabel("Recall")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center",
               ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
    paper_style.save(fig, out_dir, "fidelity_diversity")


def _node_values(feeders, key):
    return np.concatenate([np.asarray(f.attributes[key], float)[1:] for f in feeders])


def distributions(sets, out_dir):
    fig, axes = paper_style.subplots(2, 2, 3.0)
    (ax_deg, ax_len), (ax_amp, ax_ph) = axes
    names = list(STYLE)
    width = 0.8 / len(names)

    # out-degree: share of nodes per degree, 4 and above pooled
    degrees = np.arange(5)
    for k, name in enumerate(names):
        out = np.concatenate([(np.asarray(f.adjacency) != 0).sum(1) for f in sets[name]])
        share = [(np.minimum(out, 4) == d).mean() for d in degrees]
        ax_deg.bar(degrees + (k - 1) * width, share, width, color=STYLE[name][1],
                   label=STYLE[name][0])
    ax_deg.set_xticks(degrees, ["0", "1", "2", "3", "4+"])
    ax_deg.set_xlabel("Out-degree")
    ax_deg.set_ylabel("Share of nodes")

    # line length: density over log-spaced bins
    bins = np.logspace(np.log10(1e-3), np.log10(1.0), 31)
    for name in names:
        length = np.clip(_node_values(sets[name], "length"), bins[0], bins[-1])
        hist, _ = np.histogram(length, bins=bins)
        ax_len.stairs(hist / hist.sum(), bins, color=STYLE[name][1], linewidth=1.0)
    ax_len.set_xscale("log")
    ax_len.set_xticks([0.001, 0.01, 0.1, 1.0], ["0.001", "0.01", "0.1", "1"])
    ax_len.minorticks_off()
    ax_len.set_xlabel("Line length (km)")
    ax_len.set_ylabel("Share of lines")

    # ampacity: share per rating, the four common ones and the rest pooled
    common = [151.0, 165.0, 295.0, 325.0]
    for k, name in enumerate(names):
        amp = _node_values(sets[name], "ampacity")
        share = [np.isclose(amp, c).mean() for c in common]
        share.append(1.0 - sum(share))
        ax_amp.bar(np.arange(5) + (k - 1) * width, share, width, color=STYLE[name][1])
    ax_amp.set_xticks(np.arange(5), [f"{c:g}" for c in common] + ["Other"])
    ax_amp.set_xlabel("Ampacity rating (A)")
    ax_amp.set_ylabel("Share of lines")

    # phase: share per configuration, the three rare two-phase codes pooled
    groups = [("A", [0]), ("B", [1]), ("C", [2]), ("Two", [3, 4, 5]), ("ABC", [6])]
    for k, name in enumerate(names):
        phase = np.round(_node_values(sets[name], "phase")).astype(int)
        share = [np.isin(phase, codes).mean() for _, codes in groups]
        ax_ph.bar(np.arange(len(groups)) + (k - 1) * width, share, width,
                  color=STYLE[name][1])
    ax_ph.set_xticks(np.arange(len(groups)), [g for g, _ in groups])
    ax_ph.set_xlabel("Phase connection")
    ax_ph.set_ylabel("Share of lines")

    for ax in axes.flat:
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_axisbelow(True)
    fig.legend(*ax_deg.get_legend_handles_labels(), loc="outside lower center",
               ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.0)
    paper_style.save(fig, out_dir, "distributions")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=None,
                   help="comparison folder with scores.json (default: newest)")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    if args.results is None:
        root = os.path.join(REPO, "results")
        runs = sorted(d for d in os.listdir(root) if d.endswith("_model_comparison"))
        args.results = os.path.join(root, runs[-1])
    with open(os.path.join(args.results, "scores.json")) as f:
        scores = json.load(f)
    meta, results = scores["meta"], scores["results"]

    from plugins.feedergan_improved import FeederGANImproved
    from plugins.staged import build_staged

    # same draws as compare.py: pool split, then one seeded sample per model
    n, seed = meta["n"], meta["seed"]
    pool = TensorFeederSource().sample(2 * n, seed=seed)
    staged_dir = meta["checkpoints"]["staged"]
    models = {
        "staged": build_staged(os.path.join(staged_dir, "topology.pt"),
                               os.path.join(staged_dir, "attributes.pt"), device=args.device),
        "feedergan-improved": FeederGANImproved(meta["checkpoints"]["feedergan-improved"],
                                                device=args.device),
    }
    sets = {"real (floor)": pool[n:]}
    for name, gen in models.items():
        torch.manual_seed(seed)
        sets[name] = gen.sample(n)

    paper_style.apply()
    out_dir = os.path.join(args.results, "figures")
    fidelity_diversity(results, out_dir)
    distributions(sets, out_dir)


if __name__ == "__main__":
    main()
