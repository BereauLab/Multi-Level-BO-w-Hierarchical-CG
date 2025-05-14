""" Helper functions for constructing models """

from typing import List
from torch import nn


def construct_linear_layers(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    nonlinearity: nn.Module = nn.LeakyReLU(),
) -> nn.Sequential:
    """
    Constructs a torch.nn module with sequential linear layers with the specified input and
    output dimensions and the specified hidden dimensions.
    The nonlinearity is applied to all but the last layer.
    :param input_dim: Input dimension of the first layer
    :param hidden_dims: List of hidden dimensions
    :param output_dim: Output dimension of the last layer
    :param nonlinearity: Nonlinearity to apply to all but the last layer
    :return: List of linear layers
    """
    hidden_dims = [input_dim] + hidden_dims + [output_dim]
    layers = []
    for i in range(len(hidden_dims) - 1):
        linear = nn.Linear(hidden_dims[i], hidden_dims[i + 1])
        layers.append(linear)
        if i < len(hidden_dims) - 2:
            ### Add nonlinearity to all but the last layer ###
            layers.append(nonlinearity)
    return nn.Sequential(*layers)
