"""
End-to-end walkthrough of the framework using the staged reference model.

    python examples/quickstart.py --train        # train both stages, then evaluate
    python examples/quickstart.py                # evaluate existing checkpoints
    python examples/quickstart.py --category rural
    python examples/quickstart.py --linecodes path/to/Linecodes.dss   # + power flow

The point of this file is the *shape* of the code, not the numbers: notice that
nothing below refers to a GAN, a mixture model or OpenDSS.  It wires together
components that satisfy framework interfaces, so replacing any one of them is a
one-line change.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gridgen import Pipeline, catalogue                                # noqa: E402
from plugins.distribution import (                                     # noqa: E402
    AmpacityTapering, PhaseSubset, SourcePathLabeler, TensorFeederSource,
    structural_metrics)
from plugins.staged import build_staged, config                        # noqa: E402
from plugins.staged.training import train_attributes, train_topology   # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="train before evaluating")
    parser.add_argument("--iterations", type=int, default=200,
                        help="short by default so the example runs quickly")
    parser.add_argument("--graphs", type=int, default=256, help="feeders to load")
    parser.add_argument("--n", type=int, default=64, help="feeders to evaluate")
    parser.add_argument("--category", default=None)
    parser.add_argument("--linecodes", default=None, help="enables power-flow metric")
    parser.add_argument("--checkpoints", default=config.CHECKPOINT_DIR)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # 1. Data --------------------------------------------------------------
    # A DataSource yields Feeders. Categories come from a separate labeller,
    # because how a feeder is classified is independent of how it is loaded.
    source = TensorFeederSource(labeler=SourcePathLabeler(),
                                include_decomposed=False,
                                max_graphs=args.graphs)
    print(source, "|", {k: len(v) for k, v in source.by_category().items()})

    # 2. Training ----------------------------------------------------------
    topology_ckpt = os.path.join(args.checkpoints, "topology.pt")
    attribute_ckpt = os.path.join(args.checkpoints, "attributes.pt")
    if args.train:
        train_topology(source, config.TopologyConfig(iterations=args.iterations),
                       device=args.device, out_dir=args.checkpoints)
        train_attributes(source, config.AttributeConfig(iterations=args.iterations),
                         device=args.device, out_dir=args.checkpoints)
    for path in (topology_ckpt, attribute_ckpt):
        if not os.path.isfile(path):
            raise SystemExit(f"missing {path} — run once with --train")

    # 3. Generator ---------------------------------------------------------
    # Three independent stages composed into one FeederGenerator. A model that
    # generates everything at once implements FeederGenerator directly and is
    # used identically from here on -- see plugins/feedergan_improved and
    # examples/compare.py, which scores both models side by side.
    generator = build_staged(topology_ckpt, attribute_ckpt, device=args.device)
    print(generator, "| supports categories:", generator.categories or "(none)")

    # 4. Constraints and metrics ------------------------------------------
    constraints = [PhaseSubset(), AmpacityTapering()]
    metrics = structural_metrics()
    if args.linecodes:
        from plugins.distribution.powerflow import PowerFlowFeasibility
        metrics = metrics + [PowerFlowFeasibility(linecodes=args.linecodes)]

    # 5. Run ---------------------------------------------------------------
    # Constraints are calibrated on real feeders first: that is what shows
    # whether a rule is hard (real rate 1.0) or a soft tendency, and therefore
    # what target the generated feeders should be judged against.
    pipeline = Pipeline(source, generator, constraints, metrics)
    print(pipeline.run(n=args.n, category=args.category))

    print("\nregistered components:")
    for kind, names in catalogue().items():
        print(f"  {kind:<20} {', '.join(names)}")


if __name__ == "__main__":
    main()
