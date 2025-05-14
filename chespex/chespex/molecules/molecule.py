"""Module containing a molecule data structure including beads and a bonds."""

from typing import Self, Callable, Optional
import warnings
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import torch_geometric


def _node_match(n1, n2) -> bool:
    """
    Helper method for networkx to check if two nodes are equal.
    This is used for the node match function in the isomorphism
    algorithm.
    :param n1: First node.
    :param n2: Second node.
    :return: True if the nodes are equal, False otherwise.
    """
    return n1["name"] == n2["name"]


class Molecule:
    """
    Data structure for a molecule that contains a list of
    beads and a list of bonds.
    """

    def __init__(
        self,
        bead_list: list[int | str | dict],
        bond_list: list[tuple[int, int]],
    ) -> None:
        """
        Initializes the molecule.
        :param bead_list: A list of beads.
        :param bond_list: A list of bonds.
        """
        if isinstance(bead_list[0], int):
            # List of indices (convert to names)
            self.beads = [{"id": b, "name": str(b)} for b in bead_list]
        elif isinstance(bead_list[0], str):
            # List of names (use incremental indices)
            self.beads = [{"id": i, "name": b} for i, b in enumerate(bead_list)]
        elif isinstance(bead_list[0], dict):
            # List of dictionaries with bead attributes
            for i, bead in enumerate(bead_list):
                if "id" not in bead_list[i]:
                    bead["id"] = i
                if "name" not in bead_list[i]:
                    bead["name"] = str(i)
            self.beads = bead_list
        self.bonds = bond_list

    def __str__(self) -> str:
        """
        Returns a string representation of the molecule.
        :return: A string representation of the molecule.
        """
        # Sort beads
        bead_names = np.array([bead["name"] for bead in self.beads])
        permutation = np.argsort(bead_names)
        beads = " ".join(bead_names[permutation])
        # Sort bonds
        bond_permutation = {j: i for i, j in enumerate(permutation)}
        bonds = []
        for bond in self.bonds:
            bi, bj = bond_permutation[bond[0]], bond_permutation[bond[1]]
            bonds.append(f"{bi}-{bj}" if bi < bj else f"{bj}-{bi}")
        bonds = " ".join(sorted(bonds))
        return f"{beads},{bonds}"

    def as_networkx(self) -> nx.Graph:
        """
        Returns the molecule as a networkx graph.
        :return: The molecule as a networkx graph.
        """
        graph = nx.Graph()
        if len(self.beads) == 1:
            graph.add_node(0, **self.beads[0])
        else:
            graph.add_edges_from(self.bonds)
            node_attributes = {}
            for i, bead in enumerate(self.beads):
                node_attributes[i] = bead
            nx.set_node_attributes(graph, node_attributes)
        return graph

    def as_torch_graph(
        self,
        add_self_loops: bool = False,
        node_dtype: torch.dtype | None = None,
        device: str | None = None,
    ) -> Data:
        """
        Returns the molecule as a torch geometric graph.
        :param add_self_loops: Whether to add self loops to the graph.
        :param node_dtype: The data type of the node features.
        :param device: The device to put the graph on ('cpu' or 'cuda')
        :return: The molecule as a torch geometric graph.
        """
        # Create node, edge tensors from molecule
        features = [[v for k, v in bead.items() if k != "name"] for bead in self.beads]
        nodes = torch.Tensor(features)
        if node_dtype is not None:
            nodes = nodes.to(dtype=node_dtype)
        edge_index = torch.Tensor(self.bonds).long().t().contiguous()
        if len(edge_index) == 0:
            edge_index = torch.Tensor([[], []]).long()
        # Make the graph undirected if there is more than one node
        if len(self.beads) > 1:
            edge_index = torch_geometric.utils.to_undirected(edge_index)
        # Add self loops if requested
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_self_loops(
                edge_index, num_nodes=len(self.beads)
            )
        # Yield torch geometric graph object (on specific device if provided)
        if device is not None:
            return Data(nodes, edge_index).to(device)
        return Data(nodes, edge_index)

    @staticmethod
    def reconstruct(
        bead_names: list[str],
        x: torch.Tensor | list[list[float]],
        edge_index: torch.Tensor | list[list[int]],
        attribute_names: list[str] | dict[str, type],
    ) -> Self:
        """
        Reconstructs a molecule object from a torch graph and a list of corresponding bead names.
        :param bead_names: List of bead names corresponding to the node features.
        :param x: The torch graph node features. Can be a list if already converted to a list.
        :param edge_index: The torch graph edge index tensor. Can be a list if already
            converted to a list.
        :param attribute_names: The names of the node features or a dictionary
            with the names and types of the node features.
        :return: The reconstructed molecule.
        """
        # Create a list of beads from the node features
        if len(bead_names) != len(x):
            raise ValueError("Number of bead names does not match number of features.")
        beads = []
        bead_types = None
        if isinstance(attribute_names, dict):
            bead_types = list(attribute_names.values())
            attribute_names = list(attribute_names.keys())
        for i, bead_features in enumerate(x):
            bead = {"name": bead_names[i]}
            for j, attribute_name in enumerate(attribute_names):
                if torch.is_tensor(bead_features):
                    bead[attribute_name] = bead_features[j].item()
                else:
                    bead[attribute_name] = bead_features[j]
                if bead_types is not None:
                    bead[attribute_name] = bead_types[j](bead[attribute_name])
            beads.append(bead)
        # Create a list of bonds from the edge index tensor
        if torch.is_tensor(edge_index):
            bonds = edge_index.t().tolist()
        else:
            bonds = [list(i) for i in zip(*edge_index)]
        for i in range(len(bonds) - 1, -1, -1):
            if bonds[i][1] < bonds[i][0]:
                if (bonds[i][1], bonds[i][0]) not in bonds:
                    bonds.append((bonds[i][1], bonds[i][0]))
            del bonds[i]
        return Molecule(beads, sorted(bonds))

    def __eq__(self, other: Self) -> bool:
        """
        Checks if two molecules are equal, i.e., if the are isomorphic.
        :param other: The other molecule.
        :return: True if the molecules are equal, False otherwise.
        """
        return nx.is_isomorphic(
            self.as_networkx(),
            other.as_networkx(),
            node_match=_node_match,
        )

    def visualize(self, **kwargs) -> None:
        """
        Visualizes the molecule.
        """
        if "ax" in kwargs:
            graph = self.as_networkx()
            labels = nx.get_node_attributes(graph, "name")
            draw_kwargs = {
                k: v for k, v in kwargs.items() if k in nx.draw.__code__.co_varnames
            }
            nx.draw_spring(
                self.as_networkx(), labels=labels, node_color="orange", **draw_kwargs
            )
        else:
            if "figsize" not in kwargs:
                kwargs["figsize"] = (3, 3)
            figure_kwargs = {
                k: v for k, v in kwargs.items() if k in plt.figure.__code__.co_varnames
            }
            plt.figure(**figure_kwargs)
            graph = self.as_networkx()
            labels = nx.get_node_attributes(graph, "name")
            draw_kwargs = {
                k: v for k, v in kwargs.items() if k in nx.draw.__code__.co_varnames
            }
            nx.draw_spring(
                self.as_networkx(), labels=labels, node_color="orange", **draw_kwargs
            )
            plt.show()

    def write_simulation_files(
        self,
        coordinate_filename: str,
        topology_filename: Optional[str] = None,
        bead_type_mapping: Optional[dict[str, str] | Callable] = None,
        bond_lengths: Optional[dict[tuple[str, str], float]] = None,
        location: Optional[list[float] | np.ndarray] = None,
        nrexcl: Optional[int] = 3,
    ):
        """
        Writes the molecule to simulation files.
        :param coordinate_filename: The name of the coordinate file.
        :param topology_filename: The name of the topology file.
        :param bead_type_mapping: Maps between bead names and bead types. Can be
            a dictionary, a callable, or None. If None, the bead names are used as
            bead types (with '-' and '+' removed).
        :param bond_lengths: The bond lengths between different bead sizes. A dictionary
            with tuples of bead sizes (R, S, T) as keys and bond lengths as values. If
            None, the following bond lengths are used:
            {('T', 'T'): 0.29, ('T', 'S'): 0.31, ('S', 'S'): 0.33, ('T', 'R'): 0.33,
            ('S', 'R'): 0.35, ('R', 'R'): 0.38}
        :param location: This vector is added to the coordinates of the beads to place
            the ligand at a specific position in the simulation box.
        """
        if bond_lengths is None:
            bond_lengths = {
                ("T", "T"): 0.29,
                ("T", "S"): 0.31,
                ("S", "S"): 0.33,
                ("T", "R"): 0.33,
                ("S", "R"): 0.35,
                ("R", "R"): 0.38,
            }
        # Add file extensions if missing
        if topology_filename is None:
            topology_filename = coordinate_filename.replace(".gro", ".itp")
        if coordinate_filename[-4:] != ".gro":
            coordinate_filename += ".gro"
        if topology_filename[-4:] != ".itp":
            topology_filename += ".itp"
        # Filter constrainted bonds
        constrained_bonds = self.bonds.copy()
        unconstrained_bonds = []
        if len(self.beads) > 4:
            warnings.warn(
                "Overconstrained check not implemented for more than 4 beads."
            )
        if len(self.beads) == 4 and len(constrained_bonds) == 6:
            unconstrained_bonds.append(constrained_bonds.pop())
        # Generate topology file
        molecule_name = "".join([bead["name"] for bead in self.beads])
        with open(topology_filename, "w", encoding="utf-8") as topology_file:
            topology_file.write("; Generated by the CheSpEx python package\n\n")
            topology_file.write("[moleculetype]\n; molname       nrexcl\n")
            topology_file.write(f"{molecule_name}       {nrexcl}\n\n[atoms]\n")
            topology_file.write("; id type  resnr residue atom cgnr charge\n")
            for i, bead in enumerate(self.beads):
                if isinstance(bead_type_mapping, dict):
                    bead_type = bead_type_mapping[bead["name"]]
                elif callable(bead_type_mapping):
                    bead_type = bead_type_mapping(bead)
                else:
                    bead_type = bead["name"].replace("-", "").replace("+", "")
                charge = bead.get("charge", 1) - 1
                topology_file.write(f"{i+1:>4}  {bead_type:>3}    1    LIG")
                topology_file.write(f'    {bead["name"]:<4}   {i+1}   {charge:>4.1f}\n')
            topology_file.write("\n[constraints]\n;  i   j  funct  length\n")
            for bond in constrained_bonds:
                size1 = self.beads[bond[0]]["name"][0]
                size1 = "R" if size1 != "T" and size1 != "S" else size1
                size2 = self.beads[bond[1]]["name"][0]
                size2 = "R" if size2 != "T" and size2 != "S" else size2
                if (size1, size2) in bond_lengths:
                    bond_length = bond_lengths[(size1, size2)]
                else:
                    bond_length = bond_lengths[(size2, size1)]
                topology_file.write(
                    f" {bond[0]+1:>3} {bond[1]+1:>3}    1    {bond_length:.3f}\n"
                )
            if len(unconstrained_bonds) > 0:
                topology_file.write("\n[bonds]\n;  i   j  funct  length  force.c.\n")
                for bond in unconstrained_bonds:
                    size1 = self.beads[bond[0]]["name"][0]
                    size1 = "R" if size1 != "T" and size1 != "S" else size1
                    size2 = self.beads[bond[1]]["name"][0]
                    size2 = "R" if size2 != "T" and size2 != "S" else size2
                    if (size1, size2) in bond_lengths:
                        bond_length = bond_lengths[(size1, size2)]
                    else:
                        bond_length = bond_lengths[(size2, size1)]
                    topology_file.write(
                        f" {bond[0]+1:>3} {bond[1]+1:>3}    1    {bond_length:.3f}  10000.0\n"
                    )

        # Generate coordinate file
        positions = nx.spring_layout(
            self.as_networkx(), k=1, dim=3, center=(0, 0, 0), scale=0.2
        )
        with open(coordinate_filename, "w", encoding="utf-8") as coordinate_file:
            coordinate_file.write("LIG Generated by the CheSpEx python package\n")
            coordinate_file.write(f"{len(self.beads)}\n")
            for i, bead in enumerate(self.beads):
                position = positions[i]
                if location is not None:
                    position += np.array(location)
                coordinate_file.write(f'    1LIG  {bead["name"]:>5}{i+1:>5}')
                coordinate_file.write(
                    f"{position[0]:>8.3f}"
                    + f"{position[1]:>8.3f}"
                    + f"{position[2]:>8.3f}\n"
                )
            coordinate_file.write("   10.00000   10.00000   10.00000\n")
        return molecule_name
