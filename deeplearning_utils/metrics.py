import torch
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