# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Any, Dict, List, Sequence

import omegaconf
import torch
import torch.nn as nn
from solo.losses.straight import straight_loss_func
from solo.methods.base import BaseMethod
from solo.utils.misc import omegaconf_select


class Straight(BaseMethod):
    def __init__(self, cfg: omegaconf.DictConfig):
        """
        Extra cfg settings:
            method_kwargs:
                proj_output_dim (int): number of dimensions of the projected features.
                proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
                straight_loss_weight (float): weight of the straightness term.
                sim_loss_weight (float): weight of the invariance term.
                var_loss_weight (float): weight of the variance term.
                cov_loss_weight (float): weight of the covariance term.
        """

        super().__init__(cfg)

        self.straight_loss_weight: float = cfg.method_kwargs.straight_loss_weight
        self.sim_loss_weight: float = cfg.method_kwargs.sim_loss_weight
        self.var_loss_weight: float = cfg.method_kwargs.var_loss_weight
        self.cov_loss_weight: float = cfg.method_kwargs.cov_loss_weight

        proj_hidden_dim: int = cfg.method_kwargs.proj_hidden_dim
        proj_output_dim: int = cfg.method_kwargs.proj_output_dim
        
        self.sequence = cfg.method_kwargs.sequence
        self.multi_layer = cfg.method_kwargs.multi_layer
        if self.multi_layer:
            self.layers = cfg.method_kwargs.layers

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @staticmethod
    def add_and_assert_specific_cfg(cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
        """Adds method specific default values/checks for config.

        Args:
            cfg (omegaconf.DictConfig): DictConfig object.

        Returns:
            omegaconf.DictConfig: same as the argument, used to avoid errors.
        """

        cfg = super(Straight, Straight).add_and_assert_specific_cfg(cfg)

        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_output_dim")
        assert not omegaconf.OmegaConf.is_missing(cfg, "method_kwargs.proj_hidden_dim")

        cfg.method_kwargs.straight_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.straight_loss_weight",
            5.0,
        )
        cfg.method_kwargs.sim_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.sim_loss_weight",
            0.0,
        )
        cfg.method_kwargs.var_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.var_loss_weight",
            15.0,
        )
        cfg.method_kwargs.cov_loss_weight = omegaconf_select(
            cfg,
            "method_kwargs.cov_loss_weight",
            1.0,
        )

        return cfg

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"name": "projector", "params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for Straight reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of Straight loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z_list = out["z"]
        if self.multi_layer:
            mid_layers = []
            for layer in self.layers:
                mid_layers.append(out['layer%d_out' % layer])
        

        # ------- straight model loss -------
        straight_model_loss, loss_log = straight_loss_func(
            z_list, mid_layers,
            straight_loss_weight=self.straight_loss_weight,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        self.log("train_straight_model_loss", straight_model_loss, on_epoch=True, sync_dist=True)
        self.log('straight_loss', loss_log[0], on_epoch=True, sync_dist=True)
        self.log('sim_loss', loss_log[1], on_epoch=True, sync_dist=True)
        self.log('var_loss', loss_log[2], on_epoch=True, sync_dist=True)
        self.log('cov_loss', loss_log[3], on_epoch=True, sync_dist=True)

        return straight_model_loss + class_loss
