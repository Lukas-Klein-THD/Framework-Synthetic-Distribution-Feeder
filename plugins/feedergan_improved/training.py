"""
WGAN-GP training of the improved FeederGAN.

Differences from the paper's training (Algorithm 1), as in the thesis model:

    Lipschitz constraint   gradient penalty (lambda = 10) instead of weight clipping
    optimiser              Adam(0, 0.9) instead of RMSProp, same learning rate
    critic schedule        5 critic updates per generator step throughout,
                           instead of 1000 for the first 20 steps
    auxiliary losses       none

The critic sees real graphs as they are and generated graphs through the
straight-through Algorithm 2 reconstruction.  Each generated graph has as many
nodes as the real graph of the same step.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np
import torch
import torch.nn.functional as F

from gridgen.data.base import DataSource

from plugins.feedergan_improved import config
from plugins.feedergan_improved.encoding import CAT_VOCAB_SIZES, encode
from plugins.feedergan_improved.networks import Critic, Generator
from plugins.distribution.reconstruction import straight_through


def _one_hot(labels: torch.Tensor) -> torch.Tensor:
    return torch.cat([F.one_hot(labels[..., k], v).float()
                      for k, v in enumerate(CAT_VOCAB_SIZES)], dim=-1)


def _gradient_penalty(critic, real, fake):
    """WGAN-GP on lines between a real and a generated graph (all inputs jointly)."""
    b = real[0].size(0)
    eps = torch.rand(b, 1, 1, device=real[0].device)
    mixed = [(eps * r + (1.0 - eps) * f).requires_grad_(True) for r, f in zip(real, fake)]
    grads = torch.autograd.grad(critic(*mixed).sum(), mixed, create_graph=True)
    norm = torch.sqrt(sum(g.pow(2).reshape(b, -1).sum(1) for g in grads) + 1e-12)
    return ((norm - 1.0) ** 2).mean()


def train(source: DataSource,
          gen_cfg: config.GeneratorConfig | None = None,
          critic_cfg: config.CriticConfig | None = None,
          train_cfg: config.TrainConfig | None = None,
          out_dir: str | None = None, device: str = "cpu", log=print) -> str:
    """Train on ``source`` and return the path of the final checkpoint.

    ``source`` must carry the stored extras ``distance``, ``pseudo_load`` and
    ``level`` (see :class:`~plugins.distribution.data.TensorFeederSource`).
    """
    gen_cfg = gen_cfg or config.GeneratorConfig()
    critic_cfg = critic_cfg or config.CriticConfig()
    tc = train_cfg or config.TrainConfig()
    out_dir = out_dir or config.CHECKPOINT_DIR
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(tc.seed)
    rng = np.random.default_rng(tc.seed)

    # encode every real graph once
    data = []
    for i in range(len(source)):
        f = source[i]
        numeric, labels = encode(f)
        data.append((torch.as_tensor(np.asarray(f.adjacency), dtype=torch.float32),
                     torch.from_numpy(numeric), torch.from_numpy(labels)))
    log(f"encoded {len(data)} real graphs")

    def batches():
        while True:
            for start in range(0, len(data) - tc.batch_size + 1, tc.batch_size):
                idx = rng.permutation(len(data))[start:start + tc.batch_size]
                adjacency, numeric, labels = (torch.stack(t).to(device)
                                              for t in zip(*(data[i] for i in idx)))
                yield adjacency, numeric, _one_hot(labels)

    generator = Generator(gen_cfg).to(device)
    critic = Critic(critic_cfg).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=tc.lr, betas=tc.betas)
    opt_c = torch.optim.Adam(critic.parameters(), lr=tc.lr, betas=tc.betas)
    stream = batches()
    history = {"w_dist": [], "gp": [], "g_loss": []}

    def save(tag: str) -> str:
        path = os.path.join(out_dir, f"feedergan_improved_{tag}.pt")
        torch.save({"generator": generator.state_dict(), "critic": critic.state_dict(),
                    "generator_config": asdict(gen_cfg), "critic_config": asdict(critic_cfg),
                    "train_config": asdict(tc)}, path)
        return path

    for step in range(1, tc.iterations + 1):
        for _ in range(tc.n_critic):
            real_a, real_num, real_cat = next(stream)
            b, m = real_a.shape[:2]
            with torch.no_grad():
                soft, fake_num, fake_cat = generator(
                    torch.randn(b, m, gen_cfg.noise_dim, device=device))
                fake_a = straight_through(soft)
            real, fake = (real_cat, real_num, real_a), (fake_cat, fake_num, fake_a)
            d_real, d_fake = critic(*real).mean(), critic(*fake).mean()
            gp = _gradient_penalty(critic, real, fake)
            loss_c = d_fake - d_real + tc.lambda_gp * gp
            opt_c.zero_grad()
            loss_c.backward()
            opt_c.step()

        soft, fake_num, fake_cat = generator(
            torch.randn(b, m, gen_cfg.noise_dim, device=device))
        loss_g = -critic(fake_cat, fake_num, straight_through(soft)).mean()
        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        history["w_dist"].append((d_real - d_fake).item())
        history["gp"].append(gp.item())
        history["g_loss"].append(loss_g.item())
        if step % tc.checkpoint_interval == 0:
            w = float(np.mean(history["w_dist"][-tc.checkpoint_interval:]))
            log(f"step {step:>5}  W {w:+.3f}  GP {gp.item():.3f}")
            save(f"iter_{step}")
            with open(os.path.join(out_dir, "history.json"), "w") as fh:
                json.dump(history, fh)
    return save("final")
