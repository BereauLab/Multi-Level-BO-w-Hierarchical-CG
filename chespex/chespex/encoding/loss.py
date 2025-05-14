"""Loss functions for the graph autoencoder."""

from itertools import permutations
from math import nan
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_dense_batch

from chespex.molecules import Molecule


def _permute_matrices(
    permutation_indices: torch.Tensor, matrices: torch.Tensor
) -> torch.Tensor:
    """
    Permute the rows and columns of a batch of matrices according to a given batch of
    permutation indices.
    :param permutation_indices: A batch of permutation indices.
    :param matrices: A batch of matrices.
    :return: The permuted matrices.
    """
    n, k = permutation_indices.shape
    row_indices = permutation_indices.unsqueeze(-1).expand(n, k, k)
    col_indices = permutation_indices.unsqueeze(1).expand(n, k, k)
    matrix_indices = torch.arange(n).unsqueeze(-1).unsqueeze(-1).expand(n, k, k)
    permuted_matrices = matrices[matrix_indices, row_indices, col_indices]
    return permuted_matrices


def _cross_entropy_loss(
    input_nodes: torch.Tensor,
    predicted_nodes: torch.Tensor,
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    """
    Calculate the cross entropy loss for a subset of the node features.
    :param input_nodes: The input node features
        (batch size, number of nodes, number of features).
    :param predicted_nodes: The predicted node features
        (batch size, number of nodes, number of features).
    :param start_idx: The start index of the subset.
    :param end_idx: The end index of the subset.
    :return: The cross entropy loss for the subset of node features
        (batch size, number of nodes).
    """
    neg_log_prob = -F.log_softmax(predicted_nodes[:, :, start_idx:end_idx], dim=2)
    cross_entropy = torch.sum(
        input_nodes[:, :, start_idx:end_idx] * neg_log_prob, dim=2
    )
    return cross_entropy


def _get_node_loss(
    input_node_features: torch.Tensor,
    predicted_node_features: torch.Tensor,
    dummy_nodes_mask: torch.Tensor,
    n_node_classes: int,
    largest_charged_node_idx: int = 0,
) -> torch.Tensor:
    """
    Calculate the loss for the node features.
    :param input_node_features: The input node features
        (batch size, number of nodes, number of features + 1).
    :param predicted_node_features: The predicted node features
        (batch size, number of nodes, number of features  + 1).
    :param dummy_nodes_mask: A mask to ignore dummy nodes
        (batch size, number of nodes).
    :param n_node_classes: The number of node classes.
    :param largest_charged_node_class_idx: The index of the largest charged node class.
    :return: The summed loss for the node features (batch size,).
    """
    idx = (n_node_classes + 1, n_node_classes + 1 + 3, n_node_classes + 1 + 3 + 2)

    class_loss = _cross_entropy_loss(
        input_node_features, predicted_node_features, 0, idx[0]
    )
    size_loss = _cross_entropy_loss(
        input_node_features, predicted_node_features, idx[0], idx[1]
    )
    charge_loss = _cross_entropy_loss(
        input_node_features, predicted_node_features, idx[1], idx[2]
    )
    charged_mask = (
        input_node_features[:, :, : idx[0]].argmax(dim=2) <= largest_charged_node_idx
    )
    charged_mask = charged_mask * dummy_nodes_mask
    charge_loss = (charge_loss * charged_mask).sum(dim=1)
    charged_mask = charged_mask.sum(dim=1)
    charge_loss[charged_mask > 0] /= charged_mask[charged_mask > 0]
    charge_loss[charged_mask == 0] = 0
    oco_w_tfe_loss = F.mse_loss(
        input_node_features[:, :, idx[2]],
        predicted_node_features[:, :, idx[2]],
        reduction="none",
    )
    size_oco_loss = 1.5 * size_loss + oco_w_tfe_loss
    size_oco_loss = (size_oco_loss * dummy_nodes_mask).sum(
        dim=1
    ) / dummy_nodes_mask.sum(dim=1)
    return size_oco_loss + class_loss.mean(dim=1) + charge_loss


def _get_node_accuracy(
    input_node_features: torch.Tensor,
    predicted_node_features: torch.Tensor,
    permutation_indices: torch.Tensor,
    dummy_nodes_mask: torch.Tensor,
    n_node_classes: int,
    largest_charged_node_idx: int = 0,
) -> torch.Tensor:
    """
    Calculate the accuracy for the node features.
    :param input_node_features: The input node features
        (batch size, number of nodes, number of features + 1).
    :param predicted_node_features: The predicted node features
        (batch size, number of nodes, number of features + 1).
    :param permutation_indices: The permutation indices for the nodes
        (batch size, number of nodes).
    :param dummy_nodes_mask: A mask to ignore dummy nodes
        (batch size, number of nodes).
    :param n_node_classes: The number of node classes.
    :param largest_charged_node_class_idx: The index of the largest charged node class.
    :return: The accuracy for the node features.
    """
    ### Permutate the input node features ###
    n, k = permutation_indices.shape
    matrix_indices = torch.arange(n).unsqueeze(-1).expand(n, k)
    input_node_features = input_node_features[matrix_indices, permutation_indices]
    ### Calculate the accuracy for the node classes ###
    idx = (n_node_classes + 1, n_node_classes + 1 + 3, n_node_classes + 1 + 3 + 2)
    class_prediction = predicted_node_features[:, :, : idx[0]].argmax(dim=2)
    class_truth = input_node_features[:, :, : idx[0]].argmax(dim=2)
    class_accuracy = (class_prediction == class_truth).float().mean()
    size_prediction = predicted_node_features[:, :, idx[0] : idx[1]].argmax(dim=2)
    size_truth = input_node_features[:, :, idx[0] : idx[1]].argmax(dim=2)
    size_accuracy = size_prediction == size_truth
    size_accuracy = (size_accuracy * dummy_nodes_mask).sum() / dummy_nodes_mask.sum()
    charge_prediction = predicted_node_features[:, :, idx[1] : idx[2]].argmax(dim=2)
    charge_mask = (class_prediction <= largest_charged_node_idx) * dummy_nodes_mask
    charge_truth = input_node_features[:, :, idx[1] : idx[2]].argmax(dim=2)
    charge_accuracy = (charge_prediction == charge_truth).float()
    charge_accuracy = charge_accuracy[charge_mask].mean()
    return [class_accuracy.item(), size_accuracy.item(), charge_accuracy.item()]


def graph_loss(
    input_data: Data,
    node_features: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    largest_charged_node_idx: int = 0,
    return_accuracy: bool = False,
) -> torch.Tensor:
    """Calculate the loss for the graph autoencoder.
    :param input_data: The input data to the graph autoencoder.
    :param node_features: The reconstructed node features.
    :param adjacency_matrix: The reconstructed adjacency matrix.
    :param n_node_classes: The number of node classes.
    :param largest_charged_node_idx: The index of the largest charged node class.
    :param return_accuracy: Whether to return the accuracy of the autoencoder.
    :return: The loss and optionally the accuracy of the autoencoder.
    """
    n_nodes = node_features.shape[1]
    n_node_classes = input_data.x.shape[1] - 6

    ### Construct a tensor containing the adjusted input nodes ###
    extended_input_features = torch.cat(
        (
            input_data.x[:, :n_node_classes],
            torch.zeros(len(input_data.x), 1).to(node_features.device),
            input_data.x[:, n_node_classes:],
        ),
        dim=1,
    )
    input_nodes, dummy_nodes_mask = to_dense_batch(
        extended_input_features, input_data.batch, fill_value=0, max_num_nodes=n_nodes
    )
    dummy_indices = torch.where(~dummy_nodes_mask)
    dummy_indices += tuple(
        torch.ones(1, len(dummy_indices[0]), dtype=int) * n_node_classes,
    )
    input_nodes[dummy_indices] = 1

    ### Construct a tensor of the input edge information as an adjacency matrix ###
    input_adjacency_matrix = to_dense_adj(
        input_data.edge_index, input_data.batch, max_num_nodes=n_nodes
    ).to(node_features.device)

    ### Construct a mask to ignore the edges between dummy nodes ###
    edge_ignore_mask = torch.ones_like(input_adjacency_matrix)
    edge_ignore_mask[~dummy_nodes_mask] = 0
    edge_ignore_mask = edge_ignore_mask * edge_ignore_mask.transpose(1, 2)
    triu_indices = torch.triu_indices(n_nodes, n_nodes, offset=1).to(
        node_features.device
    )
    edge_ignore_mask = edge_ignore_mask[:, *triu_indices].bool()

    ### Calculate "best" node permutation and permutation invariant node loss ###
    lowest_node_loss = torch.ones(len(input_nodes)).to(node_features.device) * 1e15
    optimal_permutation = torch.arange(
        n_nodes, dtype=int, device=node_features.device
    ).repeat(len(input_nodes), 1)
    for permutation in permutations(range(n_nodes), n_nodes):
        permuted_nodes = input_nodes[:, permutation]
        node_loss = _get_node_loss(
            permuted_nodes,
            node_features,
            dummy_nodes_mask,
            n_node_classes,
            largest_charged_node_idx,
        )
        is_loss_lower = node_loss < lowest_node_loss
        optimal_permutation[is_loss_lower] = (
            torch.Tensor(permutation).long().to(node_features.device)
        )
        lowest_node_loss = torch.minimum(lowest_node_loss, node_loss)
    lowest_node_loss = lowest_node_loss.mean()

    ### Calculate the edge loss ###
    permuted_input_adjacency_matrix = _permute_matrices(
        optimal_permutation, input_adjacency_matrix
    )
    triu_indices = torch.triu_indices(n_nodes, n_nodes, offset=1).to(
        node_features.device
    )
    input_triu = permuted_input_adjacency_matrix[:, *triu_indices]
    predicted_triu = adjacency_matrix[:, *triu_indices]
    edge_loss = F.binary_cross_entropy_with_logits(
        predicted_triu, input_triu, reduction="none"
    )[edge_ignore_mask].mean()

    total_loss = 4 * edge_loss + lowest_node_loss

    if return_accuracy:
        with torch.no_grad():
            node_accuracy = _get_node_accuracy(
                input_nodes,
                node_features,
                optimal_permutation,
                dummy_nodes_mask,
                n_node_classes,
                largest_charged_node_idx,
            )
            edge_accuracy = ((predicted_triu > 0).float() == input_triu).float().mean()
            total_accuracy = [edge_accuracy.item(), *node_accuracy]
        return total_loss, np.array(total_accuracy)

    return total_loss


def prediction_to_molecules(
    node_features: torch.Tensor,
    adjacency_matrix: torch.Tensor,
    bead_class_names: list[str],
) -> list[Molecule] | Molecule:
    """
    Convert the node features and adjacency matrix of a graph to a molecule object.
    :param node_features: The node features of the graph or batch of graphs
        ([batch size,] number of nodes, number of features).
    :param adjacency_matrix: The adjacency matrix of the graph or batch of graphs
        ([batch size,] number of nodes, number of nodes).
    :param bead_class_names: The names of the bead classes.
    :return: The molecule object or list of molecule objects.
    """
    single_item = False
    if node_features.dim() == 2:
        single_item = True
        node_features = node_features.unsqueeze(0)
        adjacency_matrix = adjacency_matrix.unsqueeze(0)
    result = []
    n_node_classes = node_features.shape[2] - 6
    for i, molecule in enumerate(node_features):
        nodes = []
        for node in molecule:
            node_class = node[:n_node_classes].argmax().item()
            if node_class < n_node_classes - 1:
                ni = {
                    "id": node_class,
                    "size": node[n_node_classes : n_node_classes + 3].argmax().item(),
                    "oco_w_tfe": node[n_node_classes + 5].item(),
                }
                ni["name"] = ["", "S", "T"][ni["size"]]
                ni["name"] += bead_class_names[ni["id"]]
                if "Q" in ni["name"]:
                    ni["charge_type"] = (
                        node[n_node_classes + 3 : n_node_classes + 5].argmax().item()
                    )
                    ni["name"] += ["-", "+"][ni["charge_type"]]
                else:
                    ni["charge_type"] = nan
                nodes.append(ni)
        edges = []
        for j, row in enumerate(adjacency_matrix[i]):
            for k, edge in enumerate(row[j + 1 :], j + 1):
                if edge > 0 and j < len(nodes) and k < len(nodes):
                    edges.append((j, k))
        result.append(Molecule(nodes, edges))
    if single_item:
        return result[0]
    return result
