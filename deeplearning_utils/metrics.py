import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, List, Type, Any, Dict, Tuple, Union

def get_masks(
    batch_size: int,
    d: int,
    rng = np.random.default_rng(611),
) -> np.ndarray:
    """
    Generates binary masks of the specified shape.

    Parameters
    ----------
    batch_size : int
        Number of masks to generate.
    d : int
        Dimension of each mask.
    rng : np.random.Generator, optional
        NumPy random generator, by default a default generator is used.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(batch_size, d)` containing binary masks.

    """
    # Generate random integers between 0 and 1
    a = rng.integers(low=0, high=2, size=(batch_size, d))
    t = torch.from_numpy(a)
    return t

def reduce_tensor(
    t: torch.Tensor,
    reduction: str = "mean",
):
    if reduction == "mean":
        return t.mean(dim=0)
    elif reduction == "sum":
        return t.sum(dim=0)
    elif reduction == "none":
        return t
    else:
        return t

def masked_mse_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor = None,
    weight: tuple = None,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Computes the masked mean squared error loss.

    Parameters
    ----------
    output : torch.Tensor
        Predicted output from the model.
    target : torch.Tensor
        Ground truth target values.
    mask : torch.Tensor
        Boolean mask indicating which values to mask in the loss computation.
    weight : tuple, optional
        Weights for the loss calculation.
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean'
        | 'sum'. 'none': no reduction will be applied, 'mean': the sum
        of the output will be divided by the number of elements in the
        output, 'sum': the output will be summed.

    Returns
    -------
    torch.Tensor
        The computed weighted reduced squared error loss.

    """
    # Calculate loss only for masked values if specified
    N = len(output)
    if weight is None:
        weight = torch.ones(N)
    if mask is None:
        weight = torch.zeros(N)

    loss = weighted_mse_loss(
        output[~mask], target[~mask],
        weight=weight[~mask], reduction=reduction
    )

    return loss

def weighted_mse_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    weight: tuple = None,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Computes the weighted mean squared error loss.

    Parameters
    ----------
    output : torch.Tensor
        Predicted output from the model.
    target : torch.Tensor
        Ground truth target values.
    weight : tuple, optional
        Weights for the loss calculation,
    reduction : str, optional
        Specifies the reduction to apply to the output: 'none' | 'mean'
        | 'sum'. 'none': no reduction will be applied, 'mean': the sum
        of the output will be divided by the number of elements in the
        output, 'sum': the output will be summed.

    Returns
    -------
    torch.Tensor
        The computed weighted reduced squared error loss.

    """
    N = len(output)
    if weight is None:
        weight = torch.ones(N)
    loss = weight * torch.sum((output - target) ** 2, dim=1)

    loss = reduce_tensor(loss, reduction=reduction)

    return loss


class WeightedMSELoss(nn.Module):

    def __init__(
        self,
        use_weight: bool = False,
        use_mask: bool = False,
        reduction:str = 'none'
    ):
        super(WeightedMSELoss, self).__init__()
        self.use_weight = use_weight
        self.use_mask = use_mask
        self.reduction = reduction

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
        weight: torch.Tensor = None,
    ) -> torch.Tensor:

        if self.use_mask and mask is not None:
            loss = masked_mse_loss(
                output,
                target,
                mask=mask,
                weight=weight,
                reduction=self.reduction,
            )
        else:
            if not self.use_weight:
                weight = None
            loss = weighted_mse_loss(
                output,
                target,
                weight=weight,
                reduction=self.reduction,
            )
        return loss

    def name(self):
        name = f"WSELoss"
        if self.use_mask:
            name += "-masked"
        if self.use_weight:
            name += "-weighted"
        return name

    def __str__(self):
        return self.name()