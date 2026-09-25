from __future__ import annotations

import numpy as np
import torch
from art.attacks.evasion import ProjectedGradientDescent

from .art_pgd import ARTPGD, YoloV5ForART


class ARTMIM(ARTPGD):
    """
    Momentum Iterative Method (MIM) attack for object detection.

    Inherits from ARTPGD and configures ProjectedGradientDescent with
    momentum decay (default: decay=1.0) and num_random_init=0.
    """

    attack_mode = "detector"
    requires_targets = True
    target_format = "detector_targets"

    def __init__(
        self,
        model,
        config=None,
        target=None,
        epsilon: float = 0.031372549,
        lr: float = 0.00784313725,
        epoch: int = 5,
        decay: float = 1.0,
        img_size: int = 640,
    ):
        super().__init__(
            model=model,
            config=config,
            target=target,
            epsilon=epsilon,
            lr=lr,
            epoch=epoch,
            img_size=img_size,
        )
        self.decay = decay

        # Reconfigure attack with momentum decay and zero random initialization
        self.attack = ProjectedGradientDescent(
            estimator=self.estimator,
            norm=np.inf,
            eps=epsilon,
            eps_step=lr,
            max_iter=epoch,
            targeted=False,
            num_random_init=0,
            decay=decay,
            verbose=False,
        )