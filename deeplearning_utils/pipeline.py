import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import date
import sys, os
import time
import shutil
import pickle
import json
from typing import Optional, Callable, List, Type, Any, Dict, Tuple, Union

from .utils import (
    write_json, read_json, set_device, export_env, get_params
)
from .plots import plot_losses
from .eval import model_evaluation, from_batch_to_acc
from .train import from_batch_to_loss_supervised, train
from .utils import jsonify

class Pipeline():

    def __init__(
        self,
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
        from_batch_to_perf: Callable = from_batch_to_acc,
        batch_to_perf_kwargs: dict = {},
        from_batch_to_loss: Callable = from_batch_to_loss_supervised,
        batch_loss_params: Optional[List[Dict[str, Any]]] = [{}],
        batch_loss_params_common: Optional[Dict[str, Any]] = {},
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
        from_batch_to_perf : Callable, optional
            Function to compute performance metrics, by default from_batch_to_acc.
        batch_to_perf_kwargs : dict, optional
            Additional keyword arguments for the performance function, by default {}.
        from_batch_to_loss: Callable
            Compute the loss from a given batch
        batch_loss_params : Optional[List[Dict[str, Any]]], optional
            List of parameter dictionaries for each batch function, by default [{}].
        batch_loss_params_common : Optional[Dict[str, Any]], optional
            Common parameters for the batch functions, by default {}.
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
        """
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.seed = seed
        self.device = set_device(device)

        # Parameters for the model
        self.model_class = model_class
        self.model_params = model_params
        self.model_params_common = model_params_common

        # Parameters for the Optimizer
        self.optimizers = optimizers
        self.optim_params = optim_params
        self.optim_params_common = optim_params_common

        # Parameters for the loss function
        self.loss_fn_class = loss_fn_class
        self.loss_params = loss_params
        self.loss_params_common = loss_params_common

        # Parameters for the performance metrics, from batch to perf
        self.from_batch_to_perf = from_batch_to_perf
        self.batch_to_perf_kwargs = batch_to_perf_kwargs

        # Parameters for the training loop, from batch to loss
        self.from_batch_to_loss = from_batch_to_loss
        self.batch_loss_params = batch_loss_params
        self.batch_loss_params_common = batch_loss_params_common

        # Filenames and folders
        self.exp_name = exp_name    # Experiment name for tracking
        self.res_path = res_path    # Path to save results
        self.subfolder = subfolder  # Subfolder for saving models

        # To be initialised later:
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.info_data = {}
        self.out_fname = None
        self.requirements_fname = None
        self.model_fullpaths = []
        self.i_best_model = None

    def model_selection(
        self,
        res_path: str = 'model_',
    ) -> Tuple[torch.nn.Module, int, List[str]]:
        """
        Train and selects the best model.

        Parameters
        ----------
        res_path : str, optional
            Base name for saving the model files, by default 'model_'.

        Returns
        -------
        Tuple[torch.nn.Module]
            The best model based on validation performance
        """

        perfs_val = []              # List to store val performances for selection
        models = []                 # List to store trained models
        self.model_fullpaths = []   # List to store full path to model files

        for i in range(len(self.model_params)):
            for j in range(len(self.optim_params)):
                for k in range(len(self.loss_params)):
                    for l in range(len(self.batch_loss_params)):

                        # Get params by mixing common and specific params
                        model_param = get_params(
                            self.model_params[i], self.model_params_common
                        )
                        optim_param = get_params(
                            self.optim_params[j], self.optim_params_common
                        )
                        loss_param = get_params(
                            self.loss_params[k], self.loss_params_common
                        )
                        batch_loss_param = get_params(
                            self.batch_loss_params[l], self.batch_loss_params_common
                        )
                        optimizer_class = self.optimizers[j]

                        full_param = {
                            "model_class": self.model_class , **model_param,
                            "optimizer" : optimizer_class, **optim_param,
                            "loss_fn_class" : self.loss_fn_class, **loss_param
                        }

                        print(" --- Current parameters --- ")
                        print("\n".join([
                            f'{key}={value}' for (key, value) in full_param.items()
                        ]))
                        print(" -------------------------- ", "\n", flush=True)

                        # Set the random seed for reproducibility
                        torch.manual_seed(self.seed)
                        np.random.seed(self.seed)

                        # Instantiate model
                        model = self.model_class(**model_param)

                        # A unique name for the saved model based on current parameters
                        name_architecture = model.name()
                        model_name = f"{res_path}{name_architecture}" + "_".join([
                            f'{key}={value}' for (key, value)
                            in {
                                "optimizer" : optimizer_class, **optim_param,
                                "loss_fn_class" : self.loss_fn_class, **loss_param
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
                            model.to(device=self.device)
                        else:

                            print(f"Starting training model: {model_name}", flush=True)
                            t_start = time.time()

                            model.to(device=self.device)

                            # Instantiate optimizer
                            optimizer = optimizer_class(
                                model.parameters(),
                                **optim_param
                            )

                            # Instantiate loss
                            loss_fn = self.loss_fn_class(**loss_param)

                            # Train the model
                            losses_train, losses_val = train(
                                n_epochs=self.n_epochs,
                                optimizer=optimizer,
                                model=model,
                                loss_fn=loss_fn,
                                train_loader=self.train_loader,
                                val_loader=self.val_loader,
                                from_batch_to_loss = self.from_batch_to_loss,
                                from_batch_to_loss_kwargs = batch_loss_param,
                                device=self.device,
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
                            model,
                            self.train_loader, self.val_loader,
                            from_batch_to_perf=self.from_batch_to_perf,
                            batch_to_perf_kwargs=self.batch_to_perf_kwargs,
                            device=self.device
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
                        self.model_fullpaths.append(model_name)

        # Get the best model based on validation performance
        self.i_best_model = np.argmax(perfs_val)
        best_model = models[self.i_best_model]

        return best_model

    def run_pipeline(
        self,
        data_train,
        data_val,
        data_test,
    ):
        """
        Returns
        -------
        torch.nn.Module
            The best trained model selected based on validation performance.
        """

        np.random.seed(self.seed)

        # ResFolder/ModelClass/ModelInstance/model_name
        # Root folder for all results
        os.makedirs(f"{self.res_path}", exist_ok=True)
        # Root folder for this experiment
        os.makedirs(f"{self.res_path}{self.subfolder}", exist_ok=True)

        # -------------- Redirect output --------------
        today = str(date.today())
        self.out_fname = f'{self.res_path}{self.subfolder}{self.exp_name}-{today}'

        fout = open(self.out_fname + ".txt", 'wt')
        sys.stdout = fout
        t_start = time.time()

        # -------------- Main parameters of the experiment --------------
        print(f"Starting experience {self.exp_name}")
        print(f"Path to main folder:     {self.res_path}")
        print(f"Path to subfolder:       {self.res_path}{self.subfolder}/")
        print(f"log filename:            {self.out_fname}.txt", flush=True)
        self.requirements_fname = f"{self.out_fname}-requirements.txt"
        print(f"environment exported to: {self.requirements_fname}")
        export_env(f"{self.requirements_fname}", with_date=False)

        # Parameters that are common to all models to be trained and that then
        # "define" the experiment parameters
        common_params = {
            "seed" : self.seed, "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "model_class": self.model_class, **self.model_params_common,
            "loss_fn_class" : self.loss_fn_class, **self.loss_params_common,
            **self.optim_params_common,
            "from_batch_to_loss" : self.from_batch_to_loss,
            **self.batch_loss_params_common,
        }

        # If we gave a single optimizer, then make it a list
        if type(self.optimizers) != list:
            # And add it as a common parameter
            common_params["optimizer"] = self.optimizers
            self.optimizers = [
                self.optimizers for j in range(max(1, len(self.optim_params)))
            ]

        print("="*59)
        print(" === Common parameters === ")
        print("\n".join([
            f'{key}={value}' for (key, value) in common_params.items()
        ]))
        print("="*59, "\n", flush=True)

        # -------------- Datasets -------------
        torch.manual_seed(self.seed)
        self.train_loader = DataLoader(
            data_train, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            data_val, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(
            data_test, batch_size=self.batch_size, shuffle=True
        )

        # ----------- Model selection -----------
        best_model = self.model_selection(
            res_path=f"{self.res_path}{self.subfolder}"
        )

        # ---------  Get all info on best model  ---------
        print("="*59)
        print(f"Best model selected: {self.model_fullpaths[self.i_best_model]}", flush=True)

        # Copy all files related to the best model
        suffixes = [
            "-full_params.json", "-losses.json", "-model.pt", "-performances.json",
            "-losses.png"
        ]
        for s in suffixes:
            # From ResFolder/ModelClass/ModelInstance/model_name
            f_origin = f"{self.model_fullpaths[self.i_best_model]}{s}"
            # To ResFolder/ModelClass/self.out_fname-best-model
            f_copy = f"{self.out_fname}-best-model{s}"
            print(f"Copying {f_origin} to {f_copy}")
            shutil.copy(f_origin, f_copy)
            if ".json" in f_origin:
                json_object = read_json(f_copy)
                pretty_json = json.dumps(json_object, indent=2)
                print(pretty_json, flush=True)

        print("="*59, flush=True)

        # ----------- Evaluate best model -----------
        perf = model_evaluation(
            best_model,
            self.train_loader, self.val_loader, self.test_loader,
            from_batch_to_perf=self.from_batch_to_perf,
            batch_to_perf_kwargs=self.batch_to_perf_kwargs,
            device=self.device
        )

        # Save final performances in main folder
        write_json(perf, f"{self.out_fname}-best-model-full-performances.json")

        # --------------- End ------------------
        t_end = time.time()
        dt = t_end - t_start
        print(f"\n\nTotal execution time: {dt:.2f}", flush=True)

        # --------- Saving pipeline ------------
        self.save(f"{self.out_fname}-Pipeline.pl")
        self.save(f"{self.out_fname}-Pipeline.json")

        fout.close()

        return best_model


    def save(
        self,
        filename: str = None,
        type:str = 'pl'
    ) -> None:
        """
        Save the pipeline (either as a class or a JSON file)

        :param filename: filename of the saved pipeline, defaults to None
        :type filename: str, optional
        :param type: type of the pipeline, either as a python class (.pl)
        or as a JSON file, defaults to 'pl'
        :type type: str, optional
        """
        if filename is None:
            filename = self.out_fname + "-Pipeline"
        if type == 'json':
            class_dict = jsonify(self)
            write_json(class_dict, filename + '.json')
        else:
            with open(filename + '.pl', 'wb') as f:
                pickle.dump(self, f)

    def load(
        self,
        filename: str = None,
        path: str = '',
    ) -> None:
        """
        Load a pipeline from a Pipeline file (.pl)

        :param filename: filename of the saved pipeline, defaults to None
        :type filename: str, optional
        :param path: path to the saved pipeline, defaults to ''
        :type path: str, optional
        """
        raise NotImplementedError("pg.load not working, use pickle instead")
        with open(path + filename, 'rb') as f:
            self = pickle.load(f)