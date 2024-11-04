import matplotlib.pyplot as plt
from typing import Optional, Callable, List, Type, Any, Dict, Tuple, Union

def plot_losses(
    l_train: List[float],
    l_val: Optional[List[float]] = None,
) -> tuple:
    """
    Plots the training and validation loss over epochs.

    Parameters
    ----------
    l_train : List[float]
        List of training losses.
    l_val : Optional[List[float]], optional
        List of validation losses, by default None.

    Returns
    -------
    tuple
        A tuple containing the figure and axes objects.
    """

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True)

    # Plot training losses
    ax.plot(l_train, label='train')

    # Plot validation losses if provided
    if l_val is not None:
        ax.plot(l_val, label='validation')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    # Return the figure and axes
    return fig, ax