""" Simple & slow script to test the generation of all possible graphs for a given set of beads. """

import sys
from itertools import product
import networkx as nx
import numpy as np


def _node_match(n1, n2):
    return n1["name"] == n2["name"]


def generate(beads: list[int]) -> int:
    """
    Test function to generate all possible graphs for a given set of beads.
    This function is not intended for direct use in the project, but rather as a
    simple test to validate the optimized C++ implementation.
    """
    size = len(beads)
    graph_map = {}

    attributes = {i: {"name": v} for i, v in enumerate(beads)}

    for bonds in product([0, 1], repeat=size * (size - 1) // 2):
        adjacency = np.zeros((size, size))
        adjacency[np.triu_indices(size, k=1)] = bonds
        g = nx.from_numpy_array(adjacency)
        degrees = [val for (_, val) in g.degree()]
        if not nx.is_connected(g) or max(degrees) > 4:
            continue
        nx.set_node_attributes(g, attributes)
        edge_info = sorted(
            [f'{g.nodes[u]["name"]}-{g.nodes[v]["name"]}' for (u, v) in g.edges()]
        )
        counts = [edge_info.count(edge) for edge in set(edge_info)]
        representation = " ".join(map(str, [sum(bonds)] + sorted(degrees) + counts))
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

    result = [list(g.edges) for graphs in graph_map.values() for g in graphs]

    return len(result)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_bondsmodule.py <beads>")
        sys.exit(1)
    beads_list = list(map(int, sys.argv[1:]))
    count = generate(beads_list)
    print("Number of bond configurations:", count)
