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

import torch
import torch.nn.functional as F
from solo.utils.misc import gather
from typing import List
import random


def straightness_loss(z_list: List, gap: int = 1) -> torch.Tensor:
    """Computes the discrete curvature given batch of projected features from all views.
    Loss is -1 * curvature.
    
    Args:
        z_list (List): List of NxD Tensors containing projected features from all views.
        
    Returns:
        torch.Tensor: straightness loss.    
    """
    eps = 1e-4
    z = torch.stack(z_list, dim=1) # (B, T, m)
    z = torch.flatten(z, start_dim=2)
    outputsDiff = z[:, gap:] - z[:, :-gap]
    above = torch.einsum('bti,bti->bt', outputsDiff[:, gap:], outputsDiff[:, :-gap])
    norm = torch.norm(outputsDiff.abs() + eps, dim=-1)
    below = torch.einsum('bt,bt->bt', norm[:,gap:], norm[:,:-gap])
    straight_loss = (above / (eps + below)).mean() # avg over B and T
    
    return -1.0 * straight_loss
    
    

def invariance_loss(z_list: List) -> torch.Tensor:
    """Computes invariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    
    Args:
        z_list (List): List of NxD Tensors containing projected features from all views.
        
    Returns:
        torch.Tensor: invariance regularization loss.
    """
    z1, z2 = random.sample(z_list, 2)
    return F.mse_loss(z1, z2)


def variance_loss(z_list: List) -> torch.Tensor:
    """Computes variance loss given batch of projected features from all views.

    Args:
        z_list (List): List of NxD Tensors containing projected features from all views.

    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z = [torch.sqrt(z.var(dim=0) + eps) for z in z_list]
    std_loss = sum([torch.mean(F.relu(1 - std)) for std in std_z]) / len(z_list)
    return std_loss


def covariance_loss(z_list: List) -> torch.Tensor:
    """Computes covariance loss given batch of projected features from all views.

    Args:
        z_list (List): List of NxD Tensors containing projected features from all views.

    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z_list[0].size()

    z_zero_mean = [z - z.mean(dim=0) for z in z_list]
    cov_z_list = [z.T @ z / (N - 1) for z in z_zero_mean]

    diag = torch.eye(D, device=z_list[0].device)
    cov_loss = sum([cov_z[~diag.bool()].pow_(2).sum() / D for cov_z in cov_z_list]) / len(z_list)
    return cov_loss


def straight_loss_func(
    z_list: List,
    mid_layers: List,
    straight_loss_weight: float = 5.0,
    sim_loss_weight: float = 0.0,
    var_loss_weight: float = 15.0,
    cov_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Computes straight model loss given batch of projected features from all views.

    Args:
        z_list (List): List of NxD Tensors containing projected features from all views.
        straight_loss_weight (float): straightness loss weight.
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.

    Returns:
        torch.Tensor: straight model loss.
    """

    z_list = [gather(z) for z in z_list]
    straight_loss = straightness_loss(z_list)
    for z_mid_list in mid_layers:
        z_mid_list = [gather(z.contiguous()) for z in z_mid_list]
        straight_loss += straightness_loss(z_mid_list)
    straight_loss /= (len(mid_layers) + 1)
            
    sim_loss = invariance_loss(z_list)
    var_loss = variance_loss(z_list)
    cov_loss = covariance_loss(z_list)

    loss = straight_loss_weight * straight_loss + sim_loss_weight * sim_loss + var_loss_weight * var_loss + cov_loss_weight * cov_loss
    return loss, [straight_loss.item(), sim_loss.item(), var_loss.item(), cov_loss.item()]
