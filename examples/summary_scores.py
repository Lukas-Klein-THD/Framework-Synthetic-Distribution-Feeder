"""
Summary scores per evaluation dimension, on a 0-10 scale.

    python examples/summary_scores.py <results folder>

Reads ``gin_seeds.json`` (gin_seed_sweep.py) and ``numbers.json``
(full_results.py) and writes ``summary_scores.json``.  Realism is relative to real
vs real (10 = an independent real sample); physical and operational validity
are scored against absolute targets of 100 %, so real feeders need not reach
10 there.

    realism       mean of the structural and attributed realism scores, each
                  10 * F1(P, R) / F1(real vs real); per GIN seed, so it has a std
    physical      10 * mean(phase-valid edges, phase-valid feeders), target 100 %
    operational   10 * E * mean(V, W, U) at the paper operating point, target 100 %:
                  energised phases E gate the mean of voltage in band V, lines
                  within rating W and three-phase buses within the unbalance
                  limit U (all measured on the energised part)
    overall       arithmetic mean of the three, so one failing dimension lowers
                  the summary instead of erasing it

Within realism, precision and recall are combined with F1, and energisation
gates the operational metrics; otherwise parts and dimensions are averaged.
The per-dimension scores stay visible next to the overall one.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.dirname(os.path.abspath(__file__))]

from plugins.distribution.metrics import operational_score, physical_score  # noqa: E402
from gin_seed_sweep import add_realism_scores                               # noqa: E402

#: row name in each result file: (gin_seeds table1, numbers electrical, power-flow arm)
ROWS = {
    "real": ("real vs real", "real (reference)", "real"),
    "staged": ("staged", "staged (full)", "staged"),
    "feedergan-improved": ("feedergan-improved", "feedergan-improved", "feedergan-improved"),
}


def _stat(values) -> dict:
    values = [float(v) for v in values]
    return {"values": values, "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def summary_scores(seeds_summary: dict, numbers: dict, kw: float = 50.0) -> dict:
    t1 = add_realism_scores(seeds_summary)["table1"]
    elec, flow = numbers["electrical"], numbers["power_flow"]

    out = {}
    for label, (g, e, p) in ROWS.items():
        pf = flow[f"{kw:g} kW | {p}"]
        realism = t1[g]["score.realism"]["values"]
        physical = physical_score(elec[e]["phase-subset.edges"],
                                  elec[e]["phase-subset.feeders"])
        operational = operational_score(pf["energised"], pf["ansi_a"],
                                        1.0 - pf["overload"], pf["unbalance_ok"])
        out[label] = {
            "realism.structure": t1[g]["score.structure"],
            "realism.attributes": t1[g]["score.attributes"],
            "realism": _stat(realism),
            "physical": physical,
            "operational": operational,
            # physical and operational have no seed dimension, so the overall
            # score varies across seeds only through realism
            "overall": _stat([(r + physical + operational) / 3.0 for r in realism]),
        }
    return out


def main() -> None:
    folder = sys.argv[1]
    with open(os.path.join(folder, "gin_seeds.json")) as f:
        seeds = json.load(f)
    with open(os.path.join(folder, "numbers.json")) as f:
        numbers = json.load(f)
    scores = summary_scores(seeds["summary"], numbers)
    with open(os.path.join(folder, "summary_scores.json"), "w") as f:
        json.dump(scores, f, indent=2)
    print(f"{'':<20}{'structure':>11}{'attributes':>12}{'realism':>13}{'physical':>10}"
          f"{'operational':>13}{'overall':>13}")
    for label, s in scores.items():
        print(f"{label:<20}{s['realism.structure']['mean']:>11.2f}"
              f"{s['realism.attributes']['mean']:>12.2f}"
              f"{s['realism']['mean']:>8.2f}+-{s['realism']['std']:.2f}"
              f"{s['physical']:>10.2f}{s['operational']:>13.2f}"
              f"{s['overall']['mean']:>8.2f}+-{s['overall']['std']:.2f}")
    print(f"written to {os.path.join(folder, 'summary_scores.json')}")


if __name__ == "__main__":
    main()
