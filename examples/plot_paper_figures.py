"""
Example feeders and the stage-localisation figure, in the IEEE style.

    python examples/plot_paper_figures.py --device cuda --out <results>/figures

    example_feeders        real, staged and improved FeederGAN feeders side by side,
                           segments coloured by phase count (full text width);
                           one row, and a two-row variant
    baseline_localisation  attributed precision against recall for trained and
                           untrained components at each stage, plus the improved
                           FeederGAN (one column); numbers from full_results.py

The feeders are the first draws of the sets scored by ``compare.py`` and
``full_results.py`` (same pool split, seeds and sampling order), not picked by
hand.  Each panel states how many segments break the phase-subset rule.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paper_style                                                         # noqa: E402
from plugins.distribution import PhaseSubset, TensorFeederSource           # noqa: E402
from plugins.distribution import config as shared                          # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── feeder drawing (custom-approach postprocessing/phase_visualization.py) ───
PHASE_COUNT = np.array([1, 1, 1, 2, 2, 2, 3])          # a b c ab bc ac abc
STYLE = {3: ("#2C3E50", 0.8, 1.5), 2: ("#E67E22", 0.65, 1.2), 1: ("#16A085", 0.4, 0.9)}
LABEL = {3: "Three-phase segment", 2: "Two-phase segment", 1: "Single-phase segment"}
SUBSTATION = "#C0392B"
VIOLATION = "#8E44AD"                                   # segment breaking the phase rule
_ANGLES = [0.0, np.pi / 2, -np.pi / 2, np.pi / 4, 3 * np.pi / 4,
           -np.pi / 4, -3 * np.pi / 4, np.pi / 8, -np.pi / 8]


def algorithm4_positions(adjacency, lengths):
    """FeederGAN Algorithm 4 layout: children fan out from the parent's direction."""
    g = nx.DiGraph()
    m = adjacency.shape[0]
    g.add_nodes_from(range(m))
    g.add_edges_from(zip(*np.nonzero(adjacency)))
    desc = {n: len(nx.descendants(g, n)) for n in g.nodes()}
    rng = np.random.default_rng(0)
    pos, queue = {0: (0.0, 0.0)}, [(0, 0.0)]
    while queue:
        cur, angle = queue.pop(0)
        cx, cy = pos[cur]
        for i, child in enumerate(sorted(g.successors(cur), key=lambda c: desc[c], reverse=True)):
            a = angle + (_ANGLES[i] if i < len(_ANGLES) else rng.uniform(-np.pi, np.pi))
            step = max(0.05, float(lengths[child])) * 100.0
            pos[child] = (cx + step * np.cos(a), cy + step * np.sin(a))
            queue.append((child, a))
    return g, pos


def draw_feeder(ax, feeder):
    adjacency = (np.asarray(feeder.adjacency) != 0).astype(int)
    lengths = np.asarray(feeder.attributes["length"], float)
    counts = PHASE_COUNT[np.clip(np.asarray(feeder.attributes["phase"]).astype(int), 0, 6)]
    g, pos = algorithm4_positions(adjacency, lengths)
    for u, v in g.edges():                              # a segment carries the child's phase
        colour, width, _ = STYLE[int(counts[v])]
        (x0, y0), (x1, y1) = pos[u], pos[v]
        ax.plot([x0, x1], [y0, y1], color=colour, linewidth=width, zorder=1,
                solid_capstyle="round")
    for node, (x, y) in pos.items():
        if node:
            colour, _, size = STYLE[int(counts[node])]
            ax.plot(x, y, "o", color=colour, markersize=size, markeredgewidth=0, zorder=2)
    ax.plot(0, 0, "s", color=SUBSTATION, markersize=3.5, zorder=3)
    bad = np.flatnonzero(~PhaseSubset().check(feeder)) + 1     # check() starts at node 1
    for node in bad:
        ax.plot(*pos[int(node)], "x", color=VIOLATION, markersize=2.6,
                markeredgewidth=0.6, zorder=4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    broken = int(bad.size)
    root = {3: "three-phase", 2: "two-phase", 1: "single-phase"}[int(counts[0])]
    return broken, root


def legend_handles():
    handles = [Line2D([], [], color=STYLE[c][0], linewidth=STYLE[c][1] + 0.4, marker="o",
                      markersize=STYLE[c][2] + 1, markeredgewidth=0, label=LABEL[c])
               for c in (3, 2, 1)]
    handles.append(Line2D([], [], color=SUBSTATION, marker="s", linestyle="none",
                          markersize=3.5, label="Substation"))
    handles.append(Line2D([], [], color=VIOLATION, marker="x", linestyle="none",
                          markersize=3.5, markeredgewidth=0.8,
                          label="Segment breaking the phase rule"))
    return handles


def example_feeders(sets, out_dir, rows):
    names = [("real (floor)", "Real feeder"), ("staged", "Staged model"),
             ("feedergan-improved", "Improved FeederGAN")]
    height = 0.6 + 2.3 * rows
    fig, axes = paper_style.subplots(rows, 3, height, width_in=paper_style.TEXT_WIDTH_IN,
                                     squeeze=False)
    for r in range(rows):
        for c, (key, title) in enumerate(names):
            ax = axes[r, c]
            feeder = sets[key][r]
            broken, root = draw_feeder(ax, feeder)
            rule = ("Phase rule holds on every segment" if broken == 0
                    else f"{broken} of {feeder.num_nodes - 1} segments break the phase rule")
            heading = f"{title}\n" if r == 0 else ""
            ax.set_title(f"{heading}{root.capitalize()} substation\n{rule}", fontsize=8)
    fig.legend(handles=legend_handles(), loc="outside lower center", ncol=3, frameon=False,
               handletextpad=0.4, columnspacing=1.5, labelspacing=0.3)
    paper_style.save(fig, out_dir, "example_feeders" if rows == 1 else "example_feeders_2rows")
    plt.close(fig)


def baseline_localisation(numbers, out_dir):
    rows = numbers["baselines"]
    # marker shape = topology source, fill = attribute source
    points = [
        ("real (floor)", "Real feeders", dict(marker="o", color="0.45")),
        ("GAN + MDN (staged)", "Topology GAN and attribute model (staged)",
         dict(marker="s", color="#1f5fa8")),
        ("GAN + i.i.d. echo", "Topology GAN and independent attributes",
         dict(marker="s", facecolors="none", edgecolors="#1f5fa8")),
        ("random tree + MDN", "Random tree and attribute model",
         dict(marker="v", color="#1f5fa8")),
        ("random tree + echo", "Random tree and independent attributes",
         dict(marker="v", facecolors="none", edgecolors="#1f5fa8")),
        ("feedergan-improved", "Improved FeederGAN", dict(marker="^", color="#d1731d")),
    ]
    fig, ax = paper_style.subplots(1, 1, 3.0)
    for key, label, style in points:
        r = rows[key]
        ax.scatter(r["gin-attributed.prdc.precision"], r["gin-attributed.prdc.recall"],
                   s=22, linewidths=0.9, label=label, zorder=3, **style)
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=1,
               frameon=False, handletextpad=0.3, labelspacing=0.25)
    paper_style.save(fig, out_dir, "baseline_localisation")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="figure directory")
    p.add_argument("--numbers", default=None,
                   help="numbers.json from full_results.py (default: <out>/../numbers.json)")
    p.add_argument("--n", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from plugins.feedergan_improved import FeederGANImproved
    from plugins.staged import build_staged

    # the scored sets: same pool split, seeds and order as compare.py / full_results.py
    pool = TensorFeederSource().sample(2 * args.n, seed=args.seed)
    staged_dir = os.path.join(shared.CHECKPOINT_ROOT, "staged")
    models = {
        "staged": build_staged(os.path.join(staged_dir, "topology.pt"),
                               os.path.join(staged_dir, "attributes.pt"), device=args.device),
        "feedergan-improved": FeederGANImproved(
            os.path.join(shared.CHECKPOINT_ROOT, "feedergan_improved",
                         "feedergan_improved_final.pt"), device=args.device),
    }
    sets = {"real (floor)": pool[args.n:]}
    for name, gen in models.items():
        torch.manual_seed(args.seed)
        sets[name] = gen.sample(args.n)

    paper_style.apply()
    example_feeders(sets, args.out, rows=1)
    example_feeders(sets, args.out, rows=2)

    numbers_path = args.numbers or os.path.join(os.path.dirname(os.path.abspath(args.out)),
                                                "numbers.json")
    with open(numbers_path) as f:
        baseline_localisation(json.load(f), args.out)


if __name__ == "__main__":
    main()
