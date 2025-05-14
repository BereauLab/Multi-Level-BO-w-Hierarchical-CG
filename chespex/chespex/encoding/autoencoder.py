""" Autoencoder for molecule encoding. """

from typing import Optional
import torch
from torch import nn
from torch_geometric.data import Data

from .encoder import Encoder
from .decoder import Decoder


class Autoencoder(nn.Module):
    """
    Autoencoder for molecule encoding
    """

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        """
        Initializes the Autoencoder with an encoder and decoder module.
        :param encoder: Encoder module
        :param decoder: Decoder module
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.encoder_dim = encoder.encoder_dim
        self.latent_dim = decoder.latent_dim

    def forward(
        self, data: Data, return_latent_space: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the Autoencoder, i.e. first through the
        encoder and then through the decoder.
        :param data: Data object containing the input graph
        :return: Nodes and adjacency matrix of the reconstructed
        graph. Optionally, the latent space encoding of the graph is
        returned as well.
        """

        ### Encode data ###
        z = self.encoder(data)

        ### Decode latent space ###
        nodes, upper_adjacency_matrix = self.decoder(z)

        if return_latent_space:
            return nodes, upper_adjacency_matrix, z
        return nodes, upper_adjacency_matrix
