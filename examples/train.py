"""
Train the reference models on the same data.

    python examples/train.py --model staged --device cuda
    python examples/train.py --model feedergan-improved --device cuda

Both models read the same source: every feeder and sub-feeder with at least
M = 128 nodes, truncated to its first 128 BFS nodes.  Checkpoints go to
``checkpoints/<model>/``; the comparison uses the final checkpoint of each.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plugins.distribution import TensorFeederSource      # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["staged", "feedergan-improved"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=None,
                        help="override the configured iteration count (for smoke tests)")
    parser.add_argument("--out", default=None, help="checkpoint directory")
    parser.add_argument("--stage", default="all", choices=["all", "topology", "attributes"],
                        help="staged model only: train one stage, keep the other's checkpoint")
    args = parser.parse_args()
    start = time.time()

    if args.model == "staged":
        from dataclasses import replace
        from plugins.staged import config
        from plugins.staged.training import train_attributes, train_topology

        source = TensorFeederSource()
        print(source)
        out = args.out or config.CHECKPOINT_DIR
        topo, attr = config.TopologyConfig(), config.AttributeConfig()
        if args.iterations:
            topo = replace(topo, iterations=args.iterations)
            attr = replace(attr, iterations=args.iterations)
        if args.stage in ("all", "topology"):
            train_topology(source, topo, device=args.device, out_dir=out, log_every=500)
        if args.stage in ("all", "attributes"):
            train_attributes(source, attr, device=args.device, out_dir=out, log_every=500)
        print(f"checkpoints: {out}/topology.pt, {out}/attributes.pt")
    else:
        from dataclasses import replace
        from plugins.feedergan_improved import config
        from plugins.feedergan_improved.training import train

        source = TensorFeederSource(extras=("distance", "pseudo_load", "level"))
        print(source)
        tc = config.TrainConfig()
        if args.iterations:
            tc = replace(tc, iterations=args.iterations,
                         checkpoint_interval=min(tc.checkpoint_interval, args.iterations))
        path = train(source, train_cfg=tc, out_dir=args.out, device=args.device)
        print(f"checkpoint: {path}")
    print(f"done in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
