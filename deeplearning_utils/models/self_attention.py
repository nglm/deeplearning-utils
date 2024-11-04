import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Callable, List

from .blocks import FC_Block

class SelfAttentionModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        hidden_dims: List[int],
        out_dim: int,
        activation: Callable = F.relu,
        dropout_rate: Optional[float] = None,
        normalisation: bool = True,
    ):
        """
        Initializes the SelfAttentionModel with a multi-head attention layer and fully connected layers.

        Parameters
        ----------
        input_dim : int
            The input dimension of the model.
        num_heads : int
            The number of heads in the multi-head attention layer.
        hidden_dims : List[int]
            The hidden dimensions for the fully connected layers.
        dropout_rate : float
            The dropout rate to be applied in the fully connected layers.
        out_dim : int
            The number of output for the final layer.
        activation : Callable, optional
            The activation function to use (default is F.relu).
        dropout_rate : Optional[float], optional
            The dropout rate to be applied after the normalization. If None, no dropout is applied.
        normalisation : bool, optional
            Indicates whether to apply layer normalization (default is True).
        """
        super(SelfAttentionModel, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = out_dim
        self.activation = activation

        # Optionally use layer normalization
        self.use_normalisation = normalisation

        # Optionally use dropout
        if dropout_rate is None:
            self.use_dropout = False
        else:
            self.use_dropout = True

        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True,
        )

        # Fully connected blocks
        self.fc_blocks = []
        for hdim in hidden_dims[:-1]:
            self.fc_blocks.append(
                FC_Block(
                    input_dim, hdim,
                    activation=activation, dropout_rate=dropout_rate,
                    normalisation=normalisation
                )
            )

        # Final output layer, without bias.
        # This layer can be used to define the embeddings of each gene
        self.fc_out = nn.Linear(hidden_dims[-1], out_dim, bias=False)

        self.num_heads = num_heads

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (sequence_length, batch_size, input_dim).
        mask : torch.Tensor
            Mask tensor for the attention layer of shape (batch_size, 1, sequence_length).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, out_dim).
        """

        # Masked self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)

        # Average over the sequence length
        x = attn_output.mean(dim=0)

        # Pass through the FC blocks
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        # Output layer
        x = self.fc_out(x)

        return x

    def name(self):
        name = f"SAM-{self.num_heads}"
        for fc_block in self.fc_blocks:
            name += f"-{fc_block.name()}"
        name += f"-FC-{self.hidden_dims[-1]}x{self.output_dim}"