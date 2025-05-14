""" Decoder for the autoencoder. """

from typing import List
import numpy as np
import torch
from torch import nn
from .model_helper import construct_linear_layers


class Decoder(nn.Module):
    """
    Decoder part of the autoencoder for molecule encoding
    Reconstructs the nodes and adjacency matrix of the graph
    """

    def __init__(
        self,
        n_nodes: int,
        node_feature_dim: int,
        latent_dim: int,
        node_hidden_dims: List[int],
        encoded_node_dims: List[int],
        edge_prelayer_dims: List[int],
        edge_hidden_dims: List[int],
        nonlinearity: nn.Module = nn.LeakyReLU,
    ) -> None:
        """
        Initializes the Decoder
        :param n_nodes: Maximum number of nodes per molecule graph
        :param node_feature_dim: Number of node features
        :param latent_dim: Dimension of the latent space
        :param node_hidden_dims: Dimensions of the hidden layers of the node reconstruction
        :param encoded_node_dims: Dimensions of the hidden layers of the node encoding
        :param edge_prelayer_dims: Dimensions of the hidden layers of the edge prelayer
        :param edge_hidden_dims: Dimensions of the hidden layers of the edge reconstruction
        :param nonlinearity: Nonlinearity to use for the hidden layers
        """
        super().__init__()

        self.n_nodes = n_nodes
        self.node_feature_dim = node_feature_dim
        self.latent_dim = latent_dim

        self.edge_iterator = np.stack(np.triu_indices(n_nodes, k=1)).T

        ### Construct node reconstruction layers ###
        self.node_reconstruction = construct_linear_layers(
            latent_dim,
            node_hidden_dims,
            (node_feature_dim + 1) * n_nodes,
            nonlinearity=nonlinearity(),
        )

        ### Construct edge reconstruction layers ###
        self.reconstructed_node_encoding = construct_linear_layers(
            node_feature_dim,
            encoded_node_dims[:-1],
            encoded_node_dims[-1],
            nonlinearity=nonlinearity(),
        )
        self.edge_prelayer = construct_linear_layers(
            latent_dim,
            edge_prelayer_dims[:-1],
            edge_prelayer_dims[-1],
            nonlinearity=nonlinearity(),
        )
        self.edge_reconstruction = construct_linear_layers(
            encoded_node_dims[-1] * n_nodes + edge_prelayer_dims[-1],
            edge_hidden_dims,
            (n_nodes * (n_nodes - 1)) // 2,
            nonlinearity=nonlinearity(),
        )
        self.edge_prelayer_nonlinearity = nonlinearity()

    def forward(self, data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder
        :param data: Latent space encoding of the graph
        :return: Nodes and adjacency matrix of the reconstructed graph
        """
        ### Reconstruct nodes ###
        nodes = self.node_reconstruction(data).view(
            -1, self.n_nodes, self.node_feature_dim + 1
        )

        ### Reconstruct edges ###
        encoded_latent_space = self.edge_prelayer_nonlinearity(self.edge_prelayer(data))
        encoded_nodes = self.reconstructed_node_encoding(
            nodes.view(-1, self.node_feature_dim + 1)[:, :-1].contiguous()
        )
        encoded_nodes = encoded_nodes.view(-1, self.n_nodes * encoded_nodes.shape[-1])
        upper_adjacency_matrix = self.edge_reconstruction(
            torch.cat([encoded_nodes, encoded_latent_space], dim=-1)
        )
        adjacency_matrix = torch.zeros(data.shape[0], self.n_nodes, self.n_nodes).to(
            data.device
        )
        adjacency_matrix[:, self.edge_iterator[:, 0], self.edge_iterator[:, 1]] = (
            upper_adjacency_matrix
        )
        adjacency_matrix[:, self.edge_iterator[:, 1], self.edge_iterator[:, 0]] = (
            upper_adjacency_matrix
        )

        return nodes, adjacency_matrix
