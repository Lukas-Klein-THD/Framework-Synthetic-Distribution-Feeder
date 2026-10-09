"""
Improved FeederGAN: topology and attributes from one generator.

The second reference model, and the counterpart to :mod:`plugins.staged`.  It
follows the original FeederGAN (Liang et al., IEEE Trans. Smart Grid 2021): a
single generator emits the soft adjacency and all node attributes in one
forward pass, and a graph convolutional critic judges them together.  It
includes the changes to the critic, the attribute representation and the
training objective that make the original trainable (thesis model M1+).

Because it implements :class:`~gridgen.generation.base.FeederGenerator`
directly, the pipeline and every metric treat it exactly like the staged model.
It depends on nothing outside this package except the shared domain plugin
:mod:`plugins.distribution`.

    from plugins import feedergan_improved
    from gridgen import build

    generator = build("generator", "feedergan-improved",
                      checkpoint=..., device="cuda")

Layout
------
``config``          hyperparameters of generator, critic and training
``encoding``        attribute representation (log length, ampacity classes, ...)
``networks``        generator and critic
``generator``       the :class:`FeederGenerator` adapter
``training``        WGAN-GP training loop
"""

from plugins.feedergan_improved import config
from plugins.feedergan_improved.generator import FeederGANImproved
from plugins.feedergan_improved.networks import Critic, Generator
from plugins.feedergan_improved.training import train

__all__ = ["config", "FeederGANImproved", "Generator", "Critic", "train"]
