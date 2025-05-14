"""Module for generating simple molecule graphs."""

from typing import Iterator, Callable
import os
import warnings
import shutil
import pickle
from pathlib import Path
from itertools import product, combinations_with_replacement
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data

from chespex.molecules import bond_generator  # c-extension bond generator
from .molecule import Molecule
from .molecule import _node_match


class MoleculeGenerator:
    """
    Generator class for simple molecule graph structures.

    The molecule generator class uses a fast c-extension to generate
    valid bond configurations. This only works for max_bonds_per_bead=4
    which is the default value. If you want to use a different value,
    the much slower python implementation is used.
    """

    def __init__(
        self,
        max_beads: int,
        bead_types: list[str | dict] | str | None = None,
        max_bonds_per_bead: int = 4,
        number_of_bead_types: int | None = None,
        cache_path: str | Path = None,
    ) -> None:
        """
        Initializes the molecule generator.
        :param max_beads: The maximum number of beads in a molecule.
        :param bead_types: Available bead types for molecule generation.
            If a str or list of str is provided, the only bead type property
            is the name. If a list of dict is provided, the dict must contain
            a 'name' key and can contain additional keys for other properties.
        :param max_bonds_per_bead: The maximum number of bonds a bead
            can have.
        :param number_of_bead_types: The number of available bead types.
            This parameter is only used if bead_cluster_names is a single
            str. In this case, the bead type names are derived by appending
            the zero-based index to the string. If None, the number of bead
            types is determined from the bead_types parameter.
        :param cache_path: The path to the cache directory. Can also be
            set via the CHESPEX_CACHE environment variable. If None, the
            directory `~/.chespex/cache` in the user's home directory is used.
        """

        # Save parameters into object attributes
        self.max_beads = max_beads
        self.max_bonds_per_bead = max_bonds_per_bead
        if self.max_bonds_per_bead != 4:
            warnings.warn(
                "Since max_bonds_per_bead is not 4, the python implementation is used. "
                "This is much slower. Consider using max_bonds_per_bead=4."
            )
        self._bond_data = {}
        # Prepare bead type information
        if bead_types is None:
            bead_types = ""
        if isinstance(bead_types, str):
            # Bead names are derived by appending the zero-based index to the string
            if number_of_bead_types is None:
                raise ValueError(
                    "If bead_types is a single str, "
                    "number_of_bead_types must be provided."
                )
            self.bead_types = [
                {"name": f"{bead_types}{i}", "id": i}
                for i in range(number_of_bead_types)
            ]
        elif isinstance(bead_types, list) and isinstance(bead_types[0], dict):
            self.bead_types = bead_types
        elif isinstance(bead_types, list) and isinstance(bead_types[0], str):
            self.bead_types = [
                {"name": bead, "id": i} for i, bead in enumerate(bead_types)
            ]
        else:
            raise ValueError(
                "bead_types must be a str, a list of str, or a list of dict."
            )

        # Get cache path
        if cache_path is not None:
            self.cache_path = Path(cache_path)
        elif "CHESPEX_CACHE" in os.environ:
            self.cache_path = Path(os.environ["CHESPEX_CACHE"])
        else:
            self.cache_path = Path("~/.chespex/cache").expanduser()

    @staticmethod
    def _get_simple_beads_representation(
        beads: list[int | str],
    ) -> tuple[list[int], list[int]]:
        """
        Returns a tuple containing a simplified version of the
        bead type list and a list of indices for the correct
        ordering of the original bead list according to the
        simplified bead type list.
        :param beads: A list of beads.
        :return: (simplified bead type list, list of indices)
        """
        occurences = {}
        for i, bead in enumerate(beads):
            if bead not in occurences:
                occurences[bead] = []
            occurences[bead].append(i)
        occurence_count = [(bead, len(indices)) for bead, indices in occurences.items()]
        occurence_count.sort(key=lambda x: x[1], reverse=True)
        indices = []
        simplified_bead_list = []
        for i, (bead, _) in enumerate(occurence_count):
            simplified_bead_list.extend([i] * len(occurences[bead]))
            indices.extend(occurences[bead])
        return simplified_bead_list, indices

    def _generate_bonds_for_bead_list_python(
        self,
        simplified_bead_list: list[int],
    ) -> list[list[tuple[int, int]]]:
        """
        Generates all possible bond configurations for a given
        simplified bead type list. This is the python implementation
        of the bond generation algorithm which is much slower than
        the c-extension implementation.
        :param simplified_bead_list: A simplified list of beads, i.e. a list
            of integers that represent the different bead types.
        :return: A list of all possible bond configurations as a list of
            bond lists.
        """
        attributes = {i: {"name": v} for i, v in enumerate(simplified_bead_list)}
        size = len(simplified_bead_list)
        graph_map = {}
        for bonds in product([0, 1], repeat=size * (size - 1) // 2):
            adjacency = np.zeros((size, size))
            adjacency[np.triu_indices(size, k=1)] = bonds
            g = nx.from_numpy_array(adjacency)
            degrees = [val for (_, val) in g.degree()]
            if not nx.is_connected(g) or max(degrees) > self.max_bonds_per_bead:
                continue
            nx.set_node_attributes(g, attributes)
            edge_info = sorted(
                [f'{g.nodes[u]["name"]}-{g.nodes[v]["name"]}' for (u, v) in g.edges()]
            )
            representation = " ".join(
                map(str, [sum(bonds)] + sorted(degrees) + edge_info)
            )
            if representation not in graph_map:
                graph_map[representation] = [g]
            else:
                found = False
                for existing_graph in graph_map[representation]:
                    if nx.is_isomorphic(existing_graph, g, _node_match):
                        found = True
                        break
                if not found:
                    graph_map[representation].append(g)
        return [list(g.edges) for graphs in graph_map.values() for g in graphs]

    def _generate_bonds_for_bead_list(
        self,
        simple_repr: list[int],
    ) -> list[list[tuple[int, int]]]:
        """
        Generates all possible bond configurations for a given
        simplified bead type list. This function calls the c-extension
        implementation of the bond generation algorithm or the slower python
        variant depending on the max_bonds_per_bead parameter.
        :param simple_repr: A simplified list of beads, i.e. a list
            of integers that represent the different bead types.
        :return: A list of all possible bond configurations as a list of
            bond lists.
        """
        if self.max_bonds_per_bead == 4:
            # pylint: disable=protected-access
            return bond_generator._generate(*simple_repr)
        return self._generate_bonds_for_bead_list(simple_repr)

    def _get_bonds_for_bead_list(
        self, bead_list: list[int | str], ignore_cache: bool = False
    ) -> tuple[list[int | str], list[list[tuple[int, int]]]]:
        """
        Returns all possible bond configurations for a given
        bead list together with the bead list in a new order
        that corresponds to the indices of the bond configurations.
        Loads cached bond configurations if available.
        :param bead_list: A list of beads.
        :param ignore_cache: If True, ignores the bond cache and
            generates all bonds from scratch.
        :return: (reordered bead list, list of bond configurations)
        """
        # Obtain a simplified representation for topology recycling
        simple_repr, indices = MoleculeGenerator._get_simple_beads_representation(
            bead_list
        )
        representation = "-".join(map(str, simple_repr))
        # Reorder beads according to the simplified representation (i.e. we want our bead
        # list to correspond to the topology which we may have already generated previously)
        bead_list_reorderd = [self.bead_types[bead_list[index]] for index in indices]
        # Check if the required topology exists in the loaded bond data dict otherwise
        # generate from scratch or load from cache file
        if representation in self._bond_data:
            bond_list = self._bond_data[representation]
        else:
            if ignore_cache or len(bead_list) < 3:
                bond_list = self._generate_bonds_for_bead_list(simple_repr)
            else:
                cache_file_name = str(self.cache_path / representation)
                if os.path.isfile(cache_file_name):
                    with open(cache_file_name, "rb") as cache_file:
                        bond_list = pickle.load(cache_file)
                else:
                    bond_list = self._generate_bonds_for_bead_list(simple_repr)
                    with open(cache_file_name, "wb") as cache_file:
                        pickle.dump(bond_list, cache_file)
            if len(bead_list) <= 5:
                self._bond_data[representation] = bond_list
        return (bead_list_reorderd, bond_list)

    def generate(self, ignore_cache: bool = False) -> Iterator[Molecule]:
        """
        Generates all possible molecule graphs.
        This is the main generator function. All other generators
        use this function and convert the resulting molecules into
        other objects/representations.
        :param ignore_cache: If True, ignores the bond cache and
            generates all bonds from scratch.
        :return: An iterator over all possible molecule graphs.
        """
        # Create cache path if it doesn't exist
        if not ignore_cache:
            self.cache_path.mkdir(parents=True, exist_ok=True)
        # Generate all possible molecule graphs
        for bead_number in range(1, self.max_beads + 1):
            for bead_list in combinations_with_replacement(
                range(len(self.bead_types)), bead_number
            ):
                reorded_beads, bond_configs = self._get_bonds_for_bead_list(
                    bead_list, ignore_cache
                )
                for bond_config in bond_configs:
                    yield Molecule(reorded_beads, bond_config)

    def generate_string_representations(
        self, ignore_cache: bool = False
    ) -> Iterator[str]:
        """
        Generates all possible string representations of molecule graphs.
        :param ignore_cache: If True, ignores the bond cache and
            generates all bonds from scratch.
        :return: An iterator over all possible string representations.
        """
        for molecule in self.generate(ignore_cache):
            yield str(molecule)

    def generate_networkx_graphs(
        self, ignore_cache: bool = False
    ) -> Iterator[nx.Graph]:
        """
        Generates all possible networkx graphs of molecule graphs.
        :param ignore_cache: If True, ignores the bond cache and
            generates all bonds from scratch.
        :return: An iterator over all possible networkx graphs.
        """
        for molecule in self.generate(ignore_cache):
            yield molecule.as_networkx()

    def generate_torch_graphs(
        self,
        device: str | None = None,
        node_dtype: torch.dtype | None = None,
        ignore_cache: bool = False,
        add_self_loops: bool = False,
    ) -> Iterator[Data]:
        """
        Generates all possible torch geometric graphs of molecule graphs.
        :param device: The device to put the graphs on.
        :param node_dtype: The data type of the node features.
        :param ignore_cache: If True, ignores the bond cache and
            generates all bonds from scratch.
        :param add_self_loops: If True, adds self loops to the graphs.
        :return: An iterator over all possible torch geometric graphs.
        """
        for molecule in self.generate(ignore_cache):
            yield molecule.as_torch_graph(
                add_self_loops, node_dtype=node_dtype, device=device
            )

    def clear_cache(
        self, keep_directory: bool = True, filename_filter: Callable | None = None
    ) -> None:
        """
        Removes all files in the cache directory.
        :param keep_directory: If True, keeps the cache directory
            and only removes the files in it.
        :param filename_filter: A function that takes a filename
            and returns True if the file should be removed and
            False otherwise. If a function is given, the parent
            directory is always kept.
        """
        if keep_directory or filename_filter is not None:
            for root, _, files in os.walk(self.cache_path):
                for file in files:
                    if filename_filter is None or filename_filter(file):
                        os.remove(os.path.join(root, file))
        else:
            shutil.rmtree(self.cache_path)
