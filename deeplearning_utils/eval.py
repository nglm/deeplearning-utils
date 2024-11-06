import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import Optional, Callable, List, Type, Any, Dict, Tuple, Union

from .utils import set_device

def from_batch_to_acc(
    model,
    batch,
    device,
):

    inputs, targets = batch
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)

    # Get model predictions
    outputs = model(inputs)
    # Get the predicted class
    _, predicted = torch.max(outputs, dim=1)

    # Number of correct predictions
    correct = int((predicted == targets).sum())

    n_inputs = len(inputs)

    return correct, n_inputs

def from_batch_to_perf(
    model,
    batch,
    device,
    perf_f: Callable = F.mse_loss,
    perf_f_kwargs: dict = {"reduction" : "none"},
):

    inputs, targets = batch
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)

    # Get model predictions
    outputs = model(inputs)

    # Compute performance
    perf = perf_f(outputs, targets, **perf_f_kwargs)

    n_inputs = len(inputs)

    return perf, n_inputs

def compute_performance(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    from_batch_to_perf: Callable = from_batch_to_acc,
    batch_to_perf_kwargs : dict = {},
    device: Optional[torch.device] = None,
) -> float:
    """
    Computes the performance of the given model on the specified data loader.

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
        The performance of the model on the provided data.
    """

    model.eval()
    device = set_device(device)

    # Performance before dividing by the total number of inputs
    raw_perf = 0
    # Total number of inputs
    total = 0

    with torch.no_grad():
        for batch in loader:

            perf, n_inputs = from_batch_to_perf(
                model, batch, device, **batch_to_perf_kwargs
            )

            # Update total count and total raw perf
            raw_perf += perf
            total += n_inputs

    # Compute the mean performance
    perf = raw_perf / total
    return perf

def model_evaluation(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    from_batch_to_perf: Callable = from_batch_to_acc,
    batch_to_perf_kwargs: dict = {},
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluates the model's performance on training, val, and test datasets.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be evaluated.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation data.
    test_loader : Optional[torch.utils.data.DataLoader], optional
        DataLoader for the test data.
    device : Optional[torch.device], optional
        The device to perform the computations on, by default None.

    Returns
    -------
    Dict[str, float]
        The performance of the model on the train, validation and test dataset.
    """
    perf = {}

    # Compute performance on the training dataset
    perf["train"] = compute_performance(
        model, train_loader,
        from_batch_to_perf=from_batch_to_perf,
        batch_to_perf_kwargs=batch_to_perf_kwargs,
        device=device,
    )

    # Compute performance on the validation dataset
    perf["val"] = compute_performance(
        model, val_loader,
        from_batch_to_perf=from_batch_to_perf,
        batch_to_perf_kwargs=batch_to_perf_kwargs,
        device=device,
    )

    if test_loader is not None:
        # Compute performance on the test dataset
        perf["test"] = compute_performance(
        model, test_loader,
        from_batch_to_perf=from_batch_to_perf,
        batch_to_perf_kwargs=batch_to_perf_kwargs,
        device=device,
    )

    # Print the performances for training, validation, and test datasets
    print(f"Training performance:     {perf['train']:.4f}", flush=True)
    print(f"Validation performance:   {perf['val']:.4f}", flush=True)
    if test_loader is not None:
        print(f"Test performance:         {perf['test']:.4f}", flush=True)

    return perf