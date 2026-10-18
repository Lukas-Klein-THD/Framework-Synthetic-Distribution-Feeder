"""
Build the held-out split by source feeder.

    python examples/make_split.py                 # writes config.SPLIT_FILE
    python examples/make_split.py --test-fraction 0.25 --seed 0

A feeder and every subfeeder sampled from it always land in the same part, and
the split is stratified by category.  The test part is divided into two halves:
``reference`` (generated feeders are scored against it) and ``comparison`` (the
independent real sample of the real-vs-real row).  Models train on ``train``
only, including the profile pool of the staged model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.distribution import config                                    # noqa: E402
from plugins.distribution.data import make_split                           # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-fraction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=config.SPLIT_FILE)
    args = p.parse_args()

    split = make_split(test_fraction=args.test_fraction, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(split, f, indent=1, sort_keys=True)
    for part, c in sorted(split["counts"].items()):
        print(f"{part:<11} {c['feeders']:>5} source feeders  {c['graphs']:>5} graphs")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
