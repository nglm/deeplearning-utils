import numpy as np
import torch

import numpy as np
import torch
from typing import Tuple, Union

def split_dataset(
    dataset: Union[np.ndarray, torch.Tensor],
    train_size: float = 0.7,
    val_size: float = 0.15,
    seed: int = 611,
    rng = None,
) -> Tuple[
    Union[np.ndarray, torch.Tensor],
    Union[np.ndarray, torch.Tensor],
    Union[np.ndarray, torch.Tensor],
]:
    """
    Splits the dataset into training, validation, and test sets.

    Parameters
    ----------
    dataset : np.ndarray or torch.Tensor
        The dataset to be split.
    train_size : float, optional
        The proportion of the dataset to include in the training set, by default 0.7.
    val_size : float, optional
        The proportion of the dataset to include in the validation set, by default 0.15.
    seed : int, optional
        Seed for random number generator, by default 611.
    rng : np.random.Generator or None, optional
        Random number generator, if None, will create a new one using the seed.

    Returns
    -------
    Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]
        The train, validation, and test sets.

    Raises
    ------
    TypeError
        If the input dataset is not a numpy array or a PyTorch tensor.
    """

    total_size = len(dataset)

    if isinstance(dataset, np.ndarray):
        if rng is None:
            # Create a new random number generator using the seed
            rng = np.random.default_rng(seed)

        # Shuffle the dataset randomly for numpy array
        rng.shuffle(dataset)
    elif isinstance(dataset, torch.Tensor):
        if seed is not None:
            # Set the seed for the random number generator for PyTorch
            torch.manual_seed(seed)

        # Shuffle the dataset randomly for PyTorch tensor
        indices = torch.randperm(total_size)  # Generate random indices
        dataset = dataset[indices]  # Shuffle the dataset
    else:
        # Raise an error if the dataset type is unsupported
        raise TypeError("Input dataset must be a numpy array or a PyTorch tensor.")

    # Calculate the sizes of each split
    train_end = int(train_size * total_size)
    val_end = train_end + int(val_size * total_size)

    # Split the dataset into train, validation, and test sets
    train_set = dataset[:train_end]
    val_set = dataset[train_end:val_end]
    test_set = dataset[val_end:]

    return train_set, val_set, test_set