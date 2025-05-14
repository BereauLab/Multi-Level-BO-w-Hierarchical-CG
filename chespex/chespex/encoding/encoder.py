""" Encoder for the autoencoder. """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Set2Set
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class Encoder(nn.Module):
    """
    Encoder for molecule encoding
    """

    def __init__(
        self,
        node_feature_dim: int,
        embedding_dim: int,
        encoder_dim: int,
        message_passing_steps: int,
        latent_dim: int,
    ) -> None:
        """
        Initializes the Encoder
        :param node_feature_dim: Feature dimension of the nodes
        :param embedding_dim: Dimension of the node type embeddings
        :param encoder_dim: Dimension of the hidden layers of the encoder
        :param message_passing_steps: Number of message passing steps
        :param latent_dim: Dimension of the latent space
        """
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.encoder_dim = encoder_dim
        self.message_passing_steps = message_passing_steps

        self.embedding = nn.Linear(node_feature_dim, embedding_dim)
        self.first = nn.Linear(embedding_dim, encoder_dim)

        self.conv = SAGEConv(
            encoder_dim, encoder_dim, aggr="add", root_weight=True, project=False
        )
        self.gru = nn.GRU(encoder_dim, encoder_dim)
        self.set2set = Set2Set(encoder_dim, processing_steps=3)
        self.linear = nn.Linear(2 * encoder_dim, encoder_dim)
        self.final = nn.Linear(encoder_dim, latent_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the encoder
        :param data: Data object containing the graph(s)
        :return: Latent space encoding of the graph(s)
        """
        out = self.embedding(data.x)
        out = F.leaky_relu(out)
        out = self.first(out)
        out = F.leaky_relu(out)
        h = out.unsqueeze(0)
        for _ in range(self.message_passing_steps):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(out, data.batch)
        out = self.linear(out)
        out = F.leaky_relu(out)
        out = self.final(out)
        return out

    def encode_molecule_list(self, molecule_list: list[Data]) -> np.ndarray:
        """
        Encodes a list of molecules (torch geometric Data objects) into the latent space.
        This function should simplify the encoding during model inference.
        :param molecule_list: A list of molecules (torch geometric Data objects)
        :return: A numpy array of latent space encodings
        """
        dataloader = DataLoader(
            molecule_list, batch_size=len(molecule_list), shuffle=False
        )
        return self(next(iter(dataloader))).detach().numpy()
