import torch
import numpy as np
from typing import Optional, Callable, List, Type, Any, Dict, Tuple, Union

from .utils import set_device

def compute_accuracy(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None
) -> float:
    """
    Computes the accuracy of the given model on the specified data loader.

    Parameters
    ----------
    model : torch.nn.Module
        The (trained) model used for predictions.
    loader : torch.utils.data.DataLoader
        DataLoader that provides the input data and corresponding targets.
    device : Optional[torch.device], optional
        The device to perform the computations on, by default None.

    Returns
    -------
    float
        The accuracy of the model on the provided data.
    """

    model.eval()  # Set the model to evaluation mode
    device = set_device(device)  # Set the device for computations

    correct = 0  # Counter for correct predictions
    total = 0    # Counter for total predictions

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for contexts, targets in loader:  # Iterate through the data loader
            contexts = contexts.to(device=device)  # Move contexts to the specified device
            targets = targets.to(device=device)  # Move targets to the specified device

            outputs = model(contexts)  # Get predictions from the model
            _, predicted = torch.max(outputs, dim=1)  # Get the predicted class
            total += len(targets)  # Update total count
            correct += int((predicted == targets).sum())  # Update correct count

    acc = correct / total  # Calculate accuracy
    return acc  # Return the computed accuracy

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