import torch
from torch import nn, optim
from typing import Optional, Callable, List, Type, Any, Dict, Tuple, Union
from datetime import date, datetime

from .utils import set_device


def identity_targets(
    inputs: torch.Tensor
) -> torch.Tensor:
    """
    Creates identity targets by detaching and cloning the input tensor.

    Parameters
    ----------
    inputs : torch.Tensor
        The input tensor for which identity targets are to be created.

    Returns
    -------
    torch.Tensor
        A detached and cloned version of the input tensor.
    """

    # Detach the input tensor from the current computation graph and clone it
    return inputs.detach().clone()

def from_batch_to_loss_supervised(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(reduction="none"),
):
    inputs, targets = batch
    inputs = inputs.to(device=device)
    targets = targets.to(device=device)

    # Get model predictions
    outputs = model(inputs)

    # Calculate the loss
    loss = loss_fn(outputs, targets)

    n_inputs = len(inputs)

    return loss, n_inputs

def from_batch_to_loss_unsupervised(
    model: torch.nn.Module,
    batch: torch.Tensor,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(reduction="none"),
    get_targets: Callable = identity_targets,
):
    inputs = batch
    targets = get_targets(inputs)

    inputs = inputs.to(device=device)
    targets = targets.to(device=device)

    # Get model predictions
    outputs = model(inputs)

    # Calculate the loss
    loss = loss_fn(outputs, targets)

    n_inputs = len(inputs)

    return loss, n_inputs



def train(
    n_epochs: int,
    optimizer: torch.optim.Optimizer,
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: Optional[torch.utils.data.DataLoader] = None,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(reduction="none"),
    from_batch_to_loss: Callable = from_batch_to_loss_supervised,
    from_batch_to_loss_kwargs: dict = {},
    device: Optional[torch.device] = None
) -> List[float]:
    """
    Trains the model for a specified number of epochs.

    Parameters
    ----------
    n_epochs : int
        The number of training epochs.
    optimizer : torch.optim.Optimizer
        The optimizer used for updating model weights.
    model : torch.nn.Module
        The model to be trained.
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The loss function used to compute the loss.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training data.
    val_loader : Optional[torch.utils.data.DataLoader], optional
        DataLoader for the validation data, by default None.
    from_batch_to_loss: Callable
        Compute the loss from a given batch
    from_batch_to_loss_kwargs: dict
        Kwargs for the from_batch_to_loss function.
    device : Optional[torch.device], optional
        The device to perform the computations on, by default None.

    Returns
    -------
    List[float]
        A list containing average training losses for each epoch.
    """

    device = set_device(device)  # Set the device for computations

    # List to store training and validation losses
    losses_train = []
    losses_val = []

    optimizer.zero_grad(set_to_none=True)


    for epoch in range(1, n_epochs + 1):  # Loop over each epoch

        # ---------------------- Training phase ------------------------
        # Set the model to training mode
        model.train()

        loss_train = 0.0
        total_inputs_train = 0
        for batch in train_loader:

            loss, n_inputs = from_batch_to_loss(
                model, batch, device, loss_fn=loss_fn,
                **from_batch_to_loss_kwargs
            )
            loss.backward()

            # Update model weights
            optimizer.step()

            # Zero gradients for the next batch
            optimizer.zero_grad()

            # Accumulate loss for the epoch
            loss_train += loss.sum().item()
            total_inputs_train += n_inputs

        # Store average loss for the epoch
        losses_train.append(float(loss_train / total_inputs_train))

        # ---------------------- Evaluation phase ----------------------
        if val_loader is not None:

            # Compute validation loss
            model.eval()

            # Set the model to evaluation mode (e.g. no gradients, dropouts)
            with torch.no_grad():
                loss_val = 0.0
                total_inputs_val = 0

                for batch in val_loader:

                    loss, n_inputs = from_batch_to_loss(
                        model, batch, device, loss_fn=loss_fn,
                        **from_batch_to_loss_kwargs
                    )
                    loss_val += loss.sum().item()
                    total_inputs_val += n_inputs

            losses_val.append(float(loss_val / total_inputs_val))


        # ---------------------- Printing losses -----------------------
        # Print training loss every 5 epochs
        if epoch == 1 or epoch % 1 == 0:
            # We can now print the validation loss in addition to the training one
            print('{}  |  Epoch {}  |  Training loss {:.5f}  |  Validation loss {:.5f}'.format(
                datetime.now().time(), epoch,
                losses_train[-1], losses_val[-1]), flush=True)

    return losses_train, losses_val