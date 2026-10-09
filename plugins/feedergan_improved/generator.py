"""
The improved FeederGAN as a framework :class:`FeederGenerator`.

One forward pass produces topology and attributes together, so this model
implements the joint interface directly rather than the staged one.  Sampling:

1. draw per-node noise and run the generator;
2. reconstruct a binary graph from the soft adjacency (Algorithm 2);
3. if the result is a tree, renumber it in BFS order from its root, so it has
   the same layout as real feeders (root = node 0, parents before children);
4. decode the attributes into physical units.

Graphs that are not trees are returned unchanged.  The model declares the same
radial schema as the data, so the pipeline's schema check reports them
instead of hiding them.
"""

from __future__ import annotations

import numpy as np
import torch

from gridgen.core.feeder import Feeder, FeederSchema
from gridgen.core.registry import register
from gridgen.generation.base import FeederGenerator

from plugins.distribution import config as shared
from plugins.distribution.data import schema
from plugins.feedergan_improved.config import GeneratorConfig
from plugins.feedergan_improved.encoding import decode
from plugins.feedergan_improved.networks import Generator
from plugins.distribution.reconstruction import algorithm2, to_rooted_order


@register("generator", "feedergan-improved")
class FeederGANImproved(FeederGenerator):
    """Joint topology-and-attribute generator (thesis model M1+)."""

    def __init__(self, checkpoint: str | None = None, device: str = "cpu",
                 num_nodes: int = shared.NUM_NODES,
                 cfg: GeneratorConfig | None = None, batch_size: int = 32):
        self.device = device
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.net = Generator(cfg).to(device)
        if checkpoint:
            self.load(checkpoint)
        self.net.eval()

    def load(self, path: str) -> "FeederGANImproved":
        blob = torch.load(path, map_location=self.device, weights_only=True)
        cfg = blob["generator_config"]
        cfg["condition_on"] = {int(k): int(v) for k, v in cfg["condition_on"].items()}
        self.net = Generator(GeneratorConfig(**cfg)).to(self.device)
        self.net.load_state_dict(blob["generator"])
        self.net.eval()
        return self

    @property
    def schema(self) -> FeederSchema:
        return schema(self.num_nodes)

    @torch.no_grad()
    def sample(self, n: int, *, category: str | None = None,
               rng: np.random.Generator | None = None) -> list[Feeder]:
        self._check_category(category)                 # unconditional model
        feeders: list[Feeder] = []
        while len(feeders) < n:
            b = min(self.batch_size, n - len(feeders))
            z = torch.randn(b, self.num_nodes, self.net.cfg.noise_dim, device=self.device)
            soft, numeric, categorical = self.net(z)
            binary = algorithm2(soft).cpu().numpy()
            numeric, categorical = numeric.cpu().numpy(), categorical.cpu().numpy()
            for i in range(b):
                attributes = decode(numeric[i], categorical[i])
                adjacency, order = to_rooted_order(binary[i])
                if order is not None:
                    attributes = {k: v[order] for k, v in attributes.items()}
                feeders.append(Feeder(adjacency, attributes,
                                      {"model": "feedergan-improved",
                                       "is_tree": order is not None}))
        return feeders
