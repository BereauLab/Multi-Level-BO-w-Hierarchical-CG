"""Tests for the molecule class."""

import torch
from chespex.molecules import Molecule


def test_string_representation_1():
    """
    Test string representation of a simple molecule with one bead.
    """
    molecule = Molecule([0], [])
    assert str(molecule) == "0,"


def test_string_representation_2():
    """
    Test string representation of a simple molecule with two beads.
    """
    molecule = Molecule([0, 1], [(0, 1)])
    assert str(molecule) == "0 1,0-1"


def test_networkx_representation_1():
    """
    Test networkx representation of a simple molecule with one bead.
    """
    molecule = Molecule([0], [])
    graph = molecule.as_networkx()
    assert len(graph.nodes) == 1
    assert len(graph.edges) == 0


def test_networkx_representation_2():
    """
    Test networkx representation of a simple molecule with two beads.
    """
    molecule = Molecule([0, 1], [(0, 1)])
    graph = molecule.as_networkx()
    assert len(graph.nodes) == 2
    assert len(graph.edges) == 1
    assert graph.has_edge(0, 1)
    assert not graph.has_edge(0, 0)
    assert not graph.has_edge(1, 1)
    assert graph.nodes[0]["name"] == "0"
    assert graph.nodes[1]["name"] == "1"
    assert graph.nodes[0]["id"] == 0
    assert graph.nodes[1]["id"] == 1


def test_torch_representation_1():
    """
    Test torch geometric representation of a simple molecule with one bead.
    """
    molecule = Molecule([0], [])
    data = molecule.as_torch_graph()
    assert data.num_nodes == 1
    assert data.num_edges == 0
    assert data.x[0][0].item() == 0
    assert len(data.edge_index) == 2


def test_torch_representation_2():
    """
    Test torch geometric representation of a simple molecule with one bead and self loops.
    """
    molecule = Molecule([0], [])
    data = molecule.as_torch_graph(add_self_loops=True)
    assert data.num_nodes == 1
    assert data.num_edges == 1
    assert data.x[0] == 0
    assert (data.edge_index == torch.Tensor([[0, 0]])).all()


def test_torch_representation_3():
    """
    Test torch geometric representation of a simple molecule with two beads.
    """
    molecule = Molecule([0, 1], [(0, 1)])
    data = molecule.as_torch_graph()
    assert data.num_nodes == 2
    assert data.num_edges == 2
    assert (data.x == torch.Tensor([[0], [1]])).all()
    assert (data.edge_index == torch.Tensor([[0, 1], [1, 0]])).all()


def test_equality_1():
    """
    Test graph equality which corresponds to graph isomorphism.
    """
    molecule1 = Molecule([0, 1], [(0, 1)])
    molecule2 = Molecule([1, 0], [(1, 0)])
    assert molecule1 == molecule2


def test_equality_2():
    """
    Test graph inequality.
    """
    molecule1 = Molecule([0, 1], [(0, 1)])
    molecule2 = Molecule([0, 2], [(0, 1)])
    assert molecule1 != molecule2


def test_equality_3():
    """
    Test graph inequality.
    """
    molecule1 = Molecule([0, 1, 2], [(0, 1), (1, 2)])
    molecule2 = Molecule([0, 1, 2], [(0, 2), (1, 2)])
    assert molecule1 != molecule2


def test_equality_4():
    """
    Test graph equality for different bead orderings.
    """
    molecule1 = Molecule([0, 1, 0], [(0, 1), (1, 2)])
    molecule2 = Molecule([0, 0, 1], [(0, 2), (1, 2)])
    assert molecule1 == molecule2


def test_equality_5():
    """
    Test graph equality for different bead orderings.
    """
    molecule1 = Molecule([0, 1, 0, 2], [(0, 1), (1, 2), (2, 3)])
    molecule2 = Molecule([0, 0, 1, 2], [(0, 2), (1, 2), (1, 3)])
    assert molecule1 == molecule2
