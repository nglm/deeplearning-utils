import torch
import json
import subprocess
import copy
from datetime import date, datetime
from typing import Optional, Callable, List, Type, Any, Dict, Tuple, Union

import numpy as np
from typing import Sequence
import json

def serialize(obj):
    # To check if the object is serializable
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        if isinstance(obj, Sequence):
            res = [serialize(x) for x in obj]
        # If we are dealing with a dict of object
        elif isinstance(obj, dict):
            res = {key : serialize(item) for key, item in obj.items()}
        # If we are dealing with a dict of object
        elif isinstance(obj, np.ndarray):
            res = obj.tolist()
        else:
            res =  jsonify(obj)
        return res

def jsonify(obj):
    # Find all class property names
    property_names = [
        p for p in dir(obj.__class__)
        # None is here in case "p" is not even an attribute
        if isinstance(getattr(obj.__class__, p, None), property)
    ]
    class_dict = {}
    for p_name in property_names:
        # property value
        p_value = getattr(obj, p_name)
        # Serialize value if necessary
        class_dict[p_name] = serialize(p_value)
    return class_dict

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

def export_env(
    fname: str = "requirements.txt",
    with_date: bool = True
) -> None:
    """
    Exports the current Python environment's package requirements to a text file.

    Parameters
    ----------
    fname : str, optional
        The filename for the requirements file, by default "requirements.txt".
    with_date : bool, optional
        Indicates whether to append the current date to the filename, by default True.

    Returns
    -------
    None
    """

    if with_date:
        today = str(date.today())
        fname = f'{fname}-{today}.txt'

    with open(fname, 'w') as f:
        subprocess.run(['pip', 'freeze'], stdout=f, text=True)

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


def get_params(
    params: Union[dict, List[dict]] = {},
    common_params: dict = {},
    i: int = None,
) -> dict:
    """
    Combines common parameters with specific parameters.

    Parameters
    ----------
    params : Union[dict, List[dict]], optional
        A dictionary or a list of dictionaries with specific parameters, by default {}.
    common_params : dict, optional
        A dictionary of common parameters to be included, by default {}.
    i : int, optional
        The index of the specific parameters to use if `params` is a list, by default None.

    Returns
    -------
    dict
        A dictionary containing the combined parameters.
    """

    # Create a copy of the common parameters to avoid modifying the original
    full_params = copy.deepcopy(common_params)

    # If params is a list, update with the specific parameters at index i
    if type(params) == list:
        full_params.update(params[i])
    else:
        # Otherwise, update with the provided dictionary
        full_params.update(params)

    # Return the combined parameters
    return full_params