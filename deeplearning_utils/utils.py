import torch
import json
from typing import Optional, Callable, List, Type, Any, Dict, Tuple, Union

def write_json(dict_json: Dict[str, Any], fname: str) -> None:
    """
    Writes a dictionary to a JSON file.

    Parameters
    ----------
    dict_json : Dict[str, Any]
        The dictionary to be written to the JSON file.
    fname : str
        The filename (including path) where the JSON will be saved.

    Returns
    -------
    None
    """

    # Convert the dictionary to a JSON string with indentation
    json_str = json.dumps(dict_json, indent=2)

    # Open the specified file in write mode with UTF-8 encoding
    with open(fname, 'w', encoding='utf-8') as f:
        # Write the JSON string to the file
        f.write(json_str)

def read_json(fname: str) -> Dict[str, Any]:
    """
    Reads a JSON file and returns the corresponding dictionary.

    Parameters
    ----------
    fname : str
        The filename (including path) of the JSON file to read.

    Returns
    -------
    Dict[str, Any]
        The dictionary representation of the JSON file.
    """

    # Open the specified file in read mode with UTF-8 encoding
    with open(fname, 'r', encoding='utf-8') as json_file:
        # Load the JSON content into a Python dictionary
        json_object = json.load(json_file)

    # Return the loaded dictionary
    return json_object

def set_device(
    device: Optional[torch.device] = None
) -> torch.device:
    """
    Helper function to set the device for PyTorch operations.

    Parameters
    ----------
    device : Optional[torch.device], optional
        The device to set, by default None.

    Returns
    -------
    torch.device
        The device selected for computation (either 'cuda' or 'cpu').
    """

    if device is None:  # Check if device is not specified
        # Set device to GPU if available, otherwise fallback to CPU
        device = (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu'))
        print(f"On device {device}.")

    return device