import numpy as np
import torch

def split_dataset(dataset, train_size=0.7, val_size=0.15):
    """
    Split a given dataset (numpy array or PyTorch tensor) into train, validation, and test datasets.

    Parameters:
    - dataset: numpy array or PyTorch tensor containing the full dataset to be split
    - train_size: proportion of the dataset to include in the train split (default is 0.7)
    - val_size: proportion of the dataset to include in the validation split (default is 0.15)

    Returns:
    - train_set: training data (numpy array or PyTorch tensor)
    - val_set: validation data (numpy array or PyTorch tensor)
    - test_set: test data (numpy array or PyTorch tensor)
    """

    if isinstance(dataset, np.ndarray):
        # Shuffle the dataset randomly for numpy array
        np.random.shuffle(dataset)
    elif isinstance(dataset, torch.Tensor):
        # Shuffle the dataset randomly for PyTorch tensor
        indices = torch.randperm(dataset.size(0))  # Generate random indices
        dataset = dataset[indices]  # Shuffle the dataset
    else:
        raise TypeError("Input dataset must be a numpy array or a PyTorch tensor.")

    # Calculate the sizes of each split
    total_size = len(dataset)
    train_end = int(train_size * total_size)
    val_end = train_end + int(val_size * total_size)

    # Split the dataset into train, validation, and test sets
    train_set = dataset[:train_end]
    val_set = dataset[train_end:val_end]
    test_set = dataset[val_end:]

    return train_set, val_set, test_set