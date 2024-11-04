import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable, List

class FC_Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Callable = F.relu,
        dropout_rate: Optional[float] = None,
        normalisation: bool = True,
    ):
        """
        Initializes the FC_Block with a fully connected layer, an optional activation function,
        normalization, and dropout.

        Parameters
        ----------
        input_dim : int
            The number of input features to the fully connected layer.
        output_dim : int
            The number of output features from the fully connected layer.
        activation : Callable, optional
            The activation function to use (default is F.relu).
        dropout_rate : Optional[float], optional
            The dropout rate to be applied after the normalization. If None, no dropout is applied.
        normalisation : bool, optional
            Indicates whether to apply layer normalization (default is True).
        """

        super(FC_Block, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        # Fully connected layer
        self.fc = nn.Linear(input_dim, output_dim)

        # Optionally use layer normalization
        self.use_normalisation = normalisation
        self.layer_norm = nn.LayerNorm(output_dim)

        # Initialize dropout if a dropout rate is provided
        if dropout_rate is None:
            self.use_dropout = False
        else:
            self.dropout = nn.Dropout(dropout_rate)  # Dropout layer
            self.dropout_rate = dropout_rate
            self.use_dropout = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the FC_Block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim).
        """
        # Apply the fully connected layer followed by the activation function
        x = self.activation(self.fc(x))

        # Apply layer normalization if enabled
        if self.use_normalisation:
            x = self.layer_norm(x)

        # Apply dropout if enabled
        if self.use_dropout:
            x = self.dropout(x)

        return x

    def name(self):
        name = f"FCB-{self.input_dim}x{self.output_dim}"
        if self.use_dropout:
            name += f"-drop_{self.dropout_rate}"
        if self.use_normalisation:
            name += "-normed"
        return name