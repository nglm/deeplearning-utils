import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from datetime import date, datetime
import sys, os
import time
import shutil
import json
from typing import Optional, Callable, List, Type, Any, Dict, Tuple, Union

from .utils import write_json, read_json, set_device, export_env
from .metrics import compute_accuracy
from .plots import plot_losses

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

    return loss

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

    return loss

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

    n_batch_train = len(train_loader)
    n_batch_val = len(val_loader)

    # List to store training and validation losses
    losses_train = []
    losses_val = []

    optimizer.zero_grad(set_to_none=True)  # Zero gradients before starting training

    for epoch in range(1, n_epochs + 1):  # Loop over each epoch

        # ---------------------- Training phase ------------------------
        # Set the model to training mode
        model.train()

        loss_train = 0.0
        for batch in train_loader:

            loss = from_batch_to_loss(
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

        # Store average loss for the epoch
        losses_train.append(float(np.mean(loss_train)))

        # ---------------------- Evaluation phase ----------------------
        if val_loader is not None:

            # Compute validation loss
            model.eval()

            # Set the model to evaluation mode (e.g. no gradients, dropouts)
            with torch.no_grad():
                loss_val = 0.0

                for batch in val_loader:

                    loss = from_batch_to_loss(
                        model, batch, device, loss_fn=loss_fn,
                        **from_batch_to_loss_kwargs
                    )
                    loss_val += loss.sum().item()

            losses_val.append(float(np.mean(loss_val)))


        # ---------------------- Printing losses -----------------------
        # Print training loss every 5 epochs
        if epoch == 1 or epoch % 1 == 0:
            # We can now print the validation loss in addition to the training one
            print('{}  |  Epoch {}  |  Training loss {:.5f}  |  Validation loss {:.5f}'.format(
                datetime.now().time(), epoch,
                loss_train / n_batch_train, loss_val / n_batch_val), flush=True)

    return losses_train, losses_val


def get_model_name(
    prefix : str = "model_",
    params : dict = {},
    model = None,
) -> str:
    """
    Get a model name from its parameters

    Parameters
    ----------
    prefix : str, optional
        Prefix of the model name, by default "model_"
    params : dict, optional
        Dictionary of parameters, by default {}

    Returns
    -------
    str
        A name, describing the type of model
    """
    name = ""
    if model is not None and hasattr(model, "name"):
        name = model.name()
    else:
        name = prefix + "_".join(['%s=%s' % (k, v) for (k, v) in params.items()])
    return name

from typing import Union, List, Dict

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
    full_params = common_params.copy()

    # If params is a list, update with the specific parameters at index i
    if type(params) == list:
        full_params.update(params[i])
    else:
        # Otherwise, update with the provided dictionary
        full_params.update(params)

    # Return the combined parameters
    return full_params

def model_selection(
    model_class: Type[torch.nn.Module],
    n_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model_params: Optional[List[Dict[str, Any]]] = [{}],
    model_params_common: Optional[Dict[str, Any]] = {},
    optimizers: Optional[Tuple[List[Type[torch.optim.Optimizer]], Type[torch.optim.Optimizer]]] = optim.Adam,
    optim_params: Optional[List[Dict[str, Any]]] = [{}],
    optim_params_common: Optional[Dict[str, Any]] = {},
    loss_fn_class: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss,
    loss_params: Optional[List[Dict[str, Any]]] = [{}],
    loss_params_common: Optional[Dict[str, Any]] = {},
    performance_f: Callable = compute_accuracy,
    performance_kwargs: dict = {},
    from_batch_to_loss: Callable = from_batch_to_loss_supervised,
    from_batch_to_loss_kwargs: dict = {},
    seed: int = 611,
    res_path: str = 'model_',
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, int, List[str]]:
    """
    Train and selects the best model.

    Parameters
    ----------
    model_class : Type[torch.nn.Module]
        The class of the model to be instantiated.
    n_epochs : int
        The number of training epochs.
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training data.
    val_loader : torch.utils.data.DataLoader
        DataLoader for the validation data.
    model_params : Optional[List[Dict[str, Any]]], optional
        Parameters for the model initialization, by default [{}].
    model_params_common : Optional[Dict[str, Any]], optional
        Common parameters for the model, by default {}.
    optimizers : Optional[Tuple[List[Type[torch.optim.Optimizer]], Type[torch.optim.Optimizer]]], optional
        List of optimizer classes to use for training, by default optim.Adam.
    optim_params : Optional[List[Dict[str, Any]]], optional
        List of parameter dictionaries for each optimizer, by default [{}].
    optim_params_common : Optional[Dict[str, Any]], optional
        Common parameters for the optimizers, by default {}.
    loss_fn_class : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        The loss function used to compute the loss, by default nn.CrossEntropyLoss.
    loss_params : Optional[List[Dict[str, Any]]], optional
        List of parameter dictionaries for each loss function, by default [{}].
    loss_params_common : Optional[Dict[str, Any]], optional
        Common parameters for the loss functions, by default {}.
    performance_f : Callable, optional
        Function to compute performance metrics, by default compute_accuracy.
    performance_kwargs : dict, optional
        Additional keyword arguments for the performance function, by default {}.
    from_batch_to_loss: Callable
        Compute the loss from a given batch
    from_batch_to_loss_kwargs: dict
        Kwargs for the from_batch_to_loss function.
    seed : int, optional
        Random seed for reproducibility, by default 611.
    res_path : str, optional
        Base name for saving the model files, by default 'model_'.
    device : Optional[torch.device], optional
        The device to perform the computations on, by default None.

    Returns
    -------
    Tuple[torch.nn.Module, int, List[str]]
        The best model based on validation performance, its index in the
        list of models, and the list of model file paths.
    """

    device = set_device(device)

    # If we gave a single optimizer, then make it a list
    if type(optimizers) != list:
        optimizers = [optimizers for j in range(len(optim_params))]

    perfs_val = []         # List to store val performances for selection
    models = []            # List to store trained models
    model_fullpaths = []   # List to store full path to model files

    for i in range(len(model_params)):
        for j in range(len(optim_params)):
            for k in range(len(loss_params)):

                # Get params by mixing common and specific params
                model_param = get_params(model_params[i], model_params_common)
                optim_param = get_params(optim_params[j], optim_params_common)
                loss_param = get_params(loss_params[k], loss_params_common)
                optimizer_class = optimizer[j]

                full_param = {
                    "model_class": model_class , **model_param,
                    "optimizer" : optimizer_class, **optim_param,
                    "loss_fn_class" : loss_fn_class, **loss_param
                }

                print(" --- Current parameters --- ")
                print("\n".join([
                    f'{key}={value}' for (key, value) in full_param.items()
                ]))
                print(" -------------------------- ", "\n", flush=True)

                # Set the random seed for reproducibility
                torch.manual_seed(seed)

                # Instantiate model
                model = model_class(**model_param)

                # A unique name for the saved model based on current parameters
                name_architecture = model.name()
                model_name = f"{res_path}{name_architecture}" + "_".join([
                    f'{key}={value}' for (key, value)
                    in {
                        "optimizer" : optimizer_class, **optim_param,
                        "loss_fn_class" : loss_fn_class, **loss_param
                    }.items()
                ])

                # ResFolder/ModelClass/ModelInstance/model_name
                # With ResFolder/ModelClass/ already given by res_path
                os.makedirs(f"{res_path}{name_architecture}", exist_ok=True)

                # Don't run again if the experiment file already exists
                if os.path.isfile(f"{model_name}-model.pt"):
                    print(
                        f"Trained model already saved: {model_name}",
                        flush=True
                    )
                    # Load the trained model
                    model = torch.load(
                        f"{model_name}-model.pt", weights_only=False
                    )
                    model.to(device=device)
                else:

                    print(f"Starting training model: {model_name}", flush=True)
                    t_start = time.time()

                    model.to(device=device)

                    # Instantiate optimizer
                    optimizer = optimizer_class(
                        model.parameters(),
                        **optim_param
                    )

                    # Instantiate loss
                    loss_fn = loss_fn_class(**loss_param)

                    # Train the model
                    losses_train, losses_val = train(
                        n_epochs=n_epochs,
                        optimizer=optimizer,
                        model=model,
                        loss_fn=loss_fn,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        from_batch_to_loss = from_batch_to_loss,
                        from_batch_to_loss_kwargs = from_batch_to_loss_kwargs,
                        device=device,
                    )

                    t_end = time.time()
                    dt = t_end - t_start
                    print(f'Model trained in {dt:.2f}s', flush=True)

                    # ------------ Save basic model info ---------------
                    # Save full params as json in subfolders
                    write_json(full_param, f"{model_name}-full_params.json")

                    # Save losses as json in subfolder
                    losses_dict = {
                        "train" : losses_train,
                        "val" : losses_val,
                    }
                    write_json(losses_dict, f"{model_name}-losses.json")

                    # Save model in subfolder
                    print(
                        f"Saving trained model at: {model_name}-model.pt",
                        flush=True
                    )
                    torch.save(model, f"{model_name}-model.pt")

                # Compute training and validation performances
                perf = model_evaluation(
                    train_loader, val_loader,
                    performance_f=performance_f,
                    performance_kwargs=performance_kwargs,
                    device=device
                )

                # Save performances as json in subfolder
                write_json(perf, f"{model_name}-performances.json")

                # Make a train/val loss evolution figure
                fig, ax = plot_losses(
                    losses_dict["train"], losses_dict["val"]
                )
                ax.set_title(f"{model_name}")
                fig.savefig(f"{model_name}-losses.png")

                # Store performances and model info to return them
                perfs_val.append(perf["val"])
                models.append(model)
                model_fullpaths.append(model_name)

    # Get the best model based on validation performance
    i_best_model = np.argmax(perfs_val)
    best_model = models[i_best_model]

    return best_model, i_best_model, model_fullpaths

def model_evaluation(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: Optional[torch.utils.data.DataLoader] = None,
    performance_f: Callable = compute_accuracy,
    performance_kwargs: dict = {},
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
    perf["train"] = performance_f(
        model, train_loader, device=device, **performance_kwargs
    )

    # Compute performance on the validation dataset
    perf["val"] = performance_f(
        model, val_loader, device=device, **performance_kwargs
    )

    if test_loader is not None:
        # Compute performance on the test dataset
        perf["test"] = performance_f(
            model, test_loader, device=device, **performance_kwargs
        )

    # Print the performances for training, validation, and test datasets
    print(f"Training performance:     {perf['train']:.4f}", flush=True)
    print(f"Validation performance:   {perf['val']:.4f}", flush=True)
    if test_loader is not None:
        print(f"Test performance:         {perf['test']:.4f}", flush=True)

    return perf

def pipeline(
    data_train,
    data_val,
    data_test,
    model_class: Type[torch.nn.Module],
    batch_size: int,
    n_epochs: int,
    model_params: Optional[List[Dict[str, Any]]] = [{}],
    model_params_common: Optional[Dict[str, Any]] = {},
    optimizers: Optional[Tuple[List[Type[torch.optim.Optimizer]], Type[torch.optim.Optimizer]]] = optim.Adam,
    optim_params: Optional[List[Dict[str, Any]]] = [{}],
    optim_params_common: Optional[Dict[str, Any]] = {},
    loss_fn_class: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss,
    loss_params: Optional[List[Dict[str, Any]]] = [{}],
    loss_params_common: Optional[Dict[str, Any]] = {},
    performance_f: Callable = compute_accuracy,
    performance_kwargs: dict = {},
    from_batch_to_loss: Callable = from_batch_to_loss_supervised,
    from_batch_to_loss_kwargs: dict = {},
    seed: int = 611,
    exp_name: str = 'pipeline',
    res_path: str = './res/',
    subfolder: str = 'ModelClass/',
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """
    Executes the entire machine learning pipeline.

    Parameters
    ----------
    data_train : Dataset
        The training dataset.
    data_val : Dataset
        The validation dataset.
    data_test : Dataset
        The test dataset.
    model_class : Type[torch.nn.Module]
        The class of the model to be instantiated.
    batch_size : int
        The size of the batches for training and evaluation.
    n_epochs : int
        The number of training epochs.
    model_params : Optional[List[Dict[str, Any]]], optional
        Parameters for the model initialization, by default [{}].
    model_params_common : Optional[Dict[str, Any]], optional
        Common parameters for the model, by default {}.
    optimizers : Optional[Tuple[List[Type[torch.optim.Optimizer]], Type[torch.optim.Optimizer]]], optional
        List of optimizer classes to use for training, by default optim.Adam.
    optim_params : Optional[List[Dict[str, Any]]], optional
        List of parameter dictionaries for each optimizer, by default [{}].
    optim_params_common : Optional[Dict[str, Any]], optional
        Common parameters for the optimizers, by default {}.
    loss_fn_class : Callable[[torch.Tensor, torch.Tensor], torch.Tensor], optional
        The loss function used to compute the loss, by default nn.CrossEntropyLoss.
    loss_params : Optional[List[Dict[str, Any]]], optional
        List of parameter dictionaries for each loss function, by default [{}].
    loss_params_common : Optional[Dict[str, Any]], optional
        Common parameters for the loss functions, by default {}.
    performance_f : Callable, optional
        Function to compute performance metrics, by default compute_accuracy.
    performance_kwargs : dict, optional
        Additional keyword arguments for the performance function, by default {}.
    from_batch_to_loss: Callable
        Compute the loss from a given batch
    from_batch_to_loss_kwargs: dict
        Kwargs for the from_batch_to_loss function.
    seed : int, optional
        Random seed for reproducibility, by default 611.
    exp_name : str, optional
        Name of the experiment, by default 'pipeline'.
    res_path : str, optional
        Base path for saving results, by default './res/'.
    subfolder : str, optional
        Subfolder for saving results, by default 'ModelClass/'.
    device : Optional[torch.device], optional
        The device to perform the computations on, by default None.

    Returns
    -------
    torch.nn.Module
        The best trained model selected based on validation performance.
    """

    np.random.seed(seed)

    # ResFolder/ModelClass/ModelInstance/model_name
    # Root folder for all results
    os.makedirs(f"{res_path}", exist_ok=True)
    # Root folder for this experiment
    os.makedirs(f"{res_path}{subfolder}", exist_ok=True)

    # -------------- Redirect output --------------
    today = str(date.today())
    out_fname = f'{res_path}{subfolder}{exp_name}-{today}'

    fout = open(out_fname + ".txt", 'wt')
    sys.stdout = fout
    t_start = time.time()

    # -------------- Main parameters of the experiment --------------
    print(f"Starting experience {exp_name}")
    print(f"Path to main folder:     {res_path}")
    print(f"Path to subfolder:       {res_path}{subfolder}/")
    print(f"log filename:            {out_fname}", flush=True)
    print(f"environment exported to: {out_fname}-requirements.txt")
    export_env(f"{out_fname}-requirements.txt", with_date=False)

    device = set_device(device)

    # Parameters that are common to all models to be trained and that then
    # "define" the experiment parameters
    common_params = {
        "seed" : seed, "batch_size": batch_size, "n_epochs": n_epochs,
        "model_class": model_class, **model_params_common,
        "loss_fn_class" : loss_fn_class, **loss_params_common,
        **optim_params_common
    }

    # If we gave a single optimizer, then make it a list
    if type(optimizers) != list:
        # And add it as a common parameter
        common_params["optimizer"] = optimizers
        optimizers = [optimizers for j in range(max(1, len(optim_params)))]

    print("="*59)
    print(" === Common parameters === ")
    print("\n".join([
        f'{key}={value}' for (key, value) in common_params.items()
    ]))
    print("="*59, "\n", flush=True)

    # -------------- Datasets -------------
    torch.manual_seed(seed)
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True)

    # ----------- Model selection -----------
    best_model, i_best_model, model_fullpaths = model_selection(
        # Mandatory params
        model_class, n_epochs, train_loader, val_loader,
        # Model-related params
        model_params=model_params, model_params_common=model_params_common,
        # Params related to inputs/ouputs/targets of the model
        from_batch_to_loss = from_batch_to_loss,
        from_batch_to_loss_kwargs = from_batch_to_loss_kwargs,
        # Optimizer-related params
        optimizers=optimizers, optim_params=optim_params,
        optim_params_common=optim_params_common,
        # Loss-related params
        loss_fn_class=loss_fn_class, loss_params=loss_params,
        loss_params_common=loss_params_common,
        # Performance-related params
        performance_f=performance_f, performance_kwargs=performance_kwargs,
        # Other params
        seed=seed, res_path=f"{res_path}{subfolder}", device=device
    )

    # ---------  Get all info on best model  ---------
    print("="*59)
    print(f"Best model selected: {model_fullpaths[i_best_model]}", flush=True)

    # Copy all files related to the best model
    suffixes = [
        "-full_params.json", "-losses.json", "-model.pt", "-performances.json",
        "-losses.png"
    ]
    for s in suffixes:
        # From ResFolder/ModelClass/ModelInstance/model_name
        f_origin = f"{model_fullpaths[i_best_model]}{s}"
        # To ResFolder/ModelClass/out_fname-best-model
        f_copy = f"{out_fname}-best-model{s}"
        print(f"Copying {f_origin} to {f_copy}")
        shutil.copy(f_origin, f_copy)
        if ".json" in f_origin:
            json_object = read_json(f_copy)
            pretty_json = json.dumps(json_object, indent=2)
            print(pretty_json, flush=True)

    print("="*59, flush=True)

    # ----------- Evaluate best model -----------
    perf = model_evaluation(
        best_model, train_loader, val_loader, test_loader,
        performance_f=performance_f, performance_kwargs=performance_kwargs,
        device=device
    )

    # Save final performances in main folder
    write_json(perf, f"{out_fname}-best-model-full-performances.json")

    # --------------- End ------------------
    t_end = time.time()
    dt = t_end - t_start
    print(f"\n\nTotal execution time: {dt:.2f}", flush=True)

    fout.close()

    return best_model