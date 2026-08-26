"""
Training loops for both stages.

Kept out of the model modules so that the models stay readable as models, and so
that a new stage can reuse the data plumbing here without inheriting a training
strategy it does not want.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Callable, Sequence

import numpy as np
import torch
import torch.optim as optim

from gridgen.core.feeder import Feeder
from gridgen.data.base import DataSource

from plugins.feedergan import config
from plugins.feedergan.attributes import AttributeModel
from plugins.feedergan.profile import FeederProfile
from plugins.feedergan.topology import TreeCritic, TreeGenerator, harden


def _batches(source: DataSource, batch_size: int, seed: int = 0):
    """Infinite shuffled batches of feeders."""
    rng = np.random.default_rng(seed)
    n = len(source)
    while True:
        for start in range(0, n - batch_size + 1, batch_size):
            idx = rng.permutation(n)[start:start + batch_size]
            yield [source[int(i)] for i in idx]


def _stack_adjacency(feeders: Sequence[Feeder], device) -> torch.Tensor:
    return torch.as_tensor(np.stack([f.adjacency for f in feeders]),
                           dtype=torch.float32, device=device)


# ── Stage 1 ──────────────────────────────────────────────────────────────────
def train_topology(source: DataSource, cfg: config.TopologyConfig | None = None,
                   device: str = "cpu", out_dir: str | None = None,
                   log_every: int = 250,
                   on_log: Callable[[int, dict], None] | None = None) -> TreeGenerator:
    """WGAN-GP training of the tree generator."""
    cfg = cfg or config.TopologyConfig()
    torch.manual_seed(cfg.seed)
    out_dir = out_dir or config.CHECKPOINT_DIR
    os.makedirs(out_dir, exist_ok=True)

    generator = TreeGenerator(cfg).to(device)
    critic = TreeCritic(cfg).to(device)
    opt_g = optim.Adam(generator.parameters(), lr=cfg.lr, betas=cfg.betas)
    opt_d = optim.Adam(critic.parameters(), lr=cfg.lr, betas=cfg.betas)
    stream = _batches(source, cfg.batch_size, cfg.seed)

    for step in range(1, cfg.iterations + 1):
        tau = max(cfg.tau_end, cfg.tau_start *
                  (cfg.tau_end / cfg.tau_start) ** (step / cfg.iterations))
        generator.set_temperature(tau)

        for _ in range(cfg.n_critic):
            real = _stack_adjacency(next(stream), device)
            z = torch.randn(real.size(0), real.size(1), cfg.noise_dim, device=device)
            with torch.no_grad():
                fake = generator(z)
            penalty = _gradient_penalty(critic, real, fake, device)
            loss_d = -(critic(real).mean() - critic(fake).mean()) + cfg.lambda_gp * penalty
            opt_d.zero_grad(); loss_d.backward(); opt_d.step()

        z = torch.randn(cfg.batch_size, cfg.num_nodes, cfg.noise_dim, device=device)
        fake = generator(z)
        # weak anti-star nudge: keeps internal out-degree near one without
        # suppressing the branching real feeders need
        internal = (fake.sum(2) > 0.3).float()
        branch = ((fake.sum(2) - 1.0) ** 2 * internal).sum(1).mean()
        loss_g = -critic(fake).mean() + cfg.lambda_branch * branch
        opt_g.zero_grad(); loss_g.backward(); opt_g.step()

        if step % log_every == 0 or step == cfg.iterations:
            info = {"step": step, "loss_d": loss_d.item(), "loss_g": loss_g.item(),
                    "tau": tau}
            torch.save({"generator": generator.state_dict(), "config": asdict(cfg)},
                       os.path.join(out_dir, "topology.pt"))
            if on_log:
                on_log(step, info)
            else:
                print(f"[topology {step:>5}] d={info['loss_d']:+.4f} "
                      f"g={info['loss_g']:+.4f} tau={tau:.3f}")
    return generator


def _gradient_penalty(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    mixed = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    grads = torch.autograd.grad(critic(mixed).sum(), mixed, create_graph=True)[0]
    return ((grads.reshape(real.size(0), -1).norm(dim=1) - 1.0) ** 2).mean()


# ── Stage 2 ──────────────────────────────────────────────────────────────────
def train_attributes(source: DataSource, cfg: config.AttributeConfig | None = None,
                     device: str = "cpu", out_dir: str | None = None,
                     log_every: int = 500,
                     on_log: Callable[[int, dict], None] | None = None):
    """Maximum-likelihood training of the attribute model."""
    cfg = cfg or config.AttributeConfig()
    torch.manual_seed(cfg.seed)
    out_dir = out_dir or config.CHECKPOINT_DIR
    os.makedirs(out_dir, exist_ok=True)

    feeders = list(source)
    profile = FeederProfile.fit(feeders)
    print(profile.summary())

    model = AttributeModel(cfg).to(device)
    sample = {name: torch.as_tensor(
        np.concatenate([f.attributes[name] for f in feeders[:256]]),
        dtype=torch.float32, device=device) for name in AttributeModel.ORDER}
    model.fit(sample)
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=cfg.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.iterations,
                                                 eta_min=cfg.lr * 0.05)
    stream = _batches(source, cfg.batch_size, cfg.seed)

    for step in range(1, cfg.iterations + 1):
        batch = next(stream)
        adjacency = _stack_adjacency(batch, device)
        attributes = {name: torch.as_tensor(
            np.stack([f.attributes[name] for f in batch]),
            dtype=torch.float32, device=device) for name in AttributeModel.ORDER}
        conditioning = profile.encode(batch).to(device)     # teacher forcing

        total, parts = model.loss(adjacency, attributes, conditioning)
        opt.zero_grad(); total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step(); sched.step()

        if step % log_every == 0 or step == cfg.iterations:
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
                        "profile": profile.state(),
                        "n_classes": {n: model.heads[n].n_classes
                                      for n in AttributeModel.MARKOV}},
                       os.path.join(out_dir, "attributes.pt"))
            info = {"step": step, "loss": total.item(), **parts}
            if on_log:
                on_log(step, info)
            else:
                print(f"[attributes {step:>5}] loss={total.item():.4f} "
                      + " ".join(f"{k}={v:.3f}" for k, v in parts.items()))
    return model, profile
