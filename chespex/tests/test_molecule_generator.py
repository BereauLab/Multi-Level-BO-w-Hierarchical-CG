"""Tests for the molecule generator."""

# pylint: disable=protected-access
import os
import torch
from chespex.molecules import MoleculeGenerator


def test_bond_generation_1():
    """
    Test bond generation for a simple molecule with one bead.
    """
    beads = [0]
    expected_bonds = [[]]
    generator = MoleculeGenerator(1, ["C1"])
    generated_bonds = generator._generate_bonds_for_bead_list(beads)
    assert sorted(generated_bonds) == sorted(expected_bonds)


def test_bond_generation_2():
    """
    Test bond generation for a simple molecule with two beads.
    """
    beads = [0, 1]
    expected_bonds = [[(0, 1)]]
    generator = MoleculeGenerator(2, number_of_bead_types=2)
    generated_bonds = generator._generate_bonds_for_bead_list(beads)
    assert sorted(generated_bonds) == sorted(expected_bonds)


def test_bond_generation_3():
    """
    Test bond generation for a simple molecule with two beads.
    """
    beads = [0, 1, 2]
    expected_bonds = [
        [(0, 1), (1, 2)],
        [(0, 2), (1, 2)],
        [(0, 1), (0, 2)],
        [(0, 1), (0, 2), (1, 2)],
    ]
    generator = MoleculeGenerator(3, "C", number_of_bead_types=3)
    generated_bonds = generator._generate_bonds_for_bead_list(beads)
    assert sorted(generated_bonds) == sorted(expected_bonds)


def test_bond_generation_4(tmp_path):
    """
    Test bond cache file generation and readout for a simple molecule with three beads.
    """
    beads = [0, 2, 1]
    expected_bonds = [
        [(0, 1), (1, 2)],
        [(0, 2), (1, 2)],
        [(0, 1), (0, 2)],
        [(0, 1), (0, 2), (1, 2)],
    ]
    generator = MoleculeGenerator(3, ["C1", "C2", "C3"], cache_path=tmp_path)
    reordered_beads, generated_bonds = generator._get_bonds_for_bead_list(beads)
    assert [b["id"] for b in reordered_beads] == beads
    assert sorted(generated_bonds) == sorted(expected_bonds)
    reordered_beads, generated_bonds = generator._get_bonds_for_bead_list(beads)
    assert [b["id"] for b in reordered_beads] == beads
    assert sorted(generated_bonds) == sorted(expected_bonds)
    assert os.path.isfile(tmp_path / "0-1-2")


def test_simple_beads_representation_1():
    """
    Test simple beads representation for a simple molecule with one bead.
    """
    beads = [1]
    expected_representation = ([0], [0])
    generated_representation = MoleculeGenerator._get_simple_beads_representation(beads)
    assert generated_representation == expected_representation


def test_simple_beads_representation_2():
    """
    Test simple beads representation for a simple molecule with two beads.
    """
    beads = ["K1", "K2"]
    expected_representation = ([0, 1], [0, 1])
    generated_representation = MoleculeGenerator._get_simple_beads_representation(beads)
    assert generated_representation == expected_representation


def test_simple_beads_representation_3():
    """
    Test simple beads representation for a simple molecule with three beads.
    """
    beads = ["K1", "K3", "K2"]
    expected_representation = ([0, 1, 2], [0, 1, 2])
    generated_representation = MoleculeGenerator._get_simple_beads_representation(beads)
    assert generated_representation == expected_representation


def test_simple_beads_representation_4():
    """
    Test simple beads representation for a molecule with four beads.
    """
    beads = [1, 2, 3, 1]
    expected_representation = ([0, 0, 1, 2], [0, 3, 1, 2])
    generated_representation = MoleculeGenerator._get_simple_beads_representation(beads)
    assert generated_representation == expected_representation


def test_simple_beads_representation_5():
    """
    Test simple beads representation for a simple molecule with three beads.
    """
    beads = [1, 2, 2, 1, 2]
    expected_representation = ([0, 0, 0, 1, 1], [1, 2, 4, 0, 3])
    generated_representation = MoleculeGenerator._get_simple_beads_representation(beads)
    assert generated_representation == expected_representation


def test_molecule_generation_1():
    """
    Test molecule generation for two-bead molecules with three different bead types.
    """
    expected_beads = [
        [0],
        [1],
        [2],  # Single bead molecules
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 1],
        [1, 2],
        [2, 2],  # Two bead molecules
    ]
    expected_bonds = [
        [],
        [],
        [],
        [(0, 1)],
        [(0, 1)],
        [(0, 1)],
        [(0, 1)],
        [(0, 1)],
        [(0, 1)],
    ]
    molecule_generator = MoleculeGenerator(2, ["C1", "C2", "C3"])
    for i, molecule in enumerate(molecule_generator.generate()):
        assert [b["id"] for b in molecule.beads] == expected_beads[i]
        assert molecule.bonds == expected_bonds[i]


def test_molecule_generation_2():
    """
    Test molecule generation for two-bead molecules with three different bead types.
    """
    expected_beads = [
        ["K1"],
        ["K2"],
        ["K3"],  # Single bead molecules
        ["K1", "K1"],
        ["K1", "K2"],
        ["K1", "K3"],  # Two bead molecules (part 1)
        ["K2", "K2"],
        ["K2", "K3"],
        ["K3", "K3"],  # Two bead molecules (part 2)
    ]
    molecule_generator = MoleculeGenerator(2, ["K1", "K2", "K3"])
    for i, molecule in enumerate(molecule_generator.generate()):
        assert [b["name"] for b in molecule.beads] == expected_beads[i]


def test_molecule_generation_3():
    """
    Test molecule generation for five-bead molecules with seven different bead types.
    """
    bead_cluster = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
    molecule_generator = MoleculeGenerator(5, bead_cluster)
    molecule_list = list(molecule_generator.generate())
    assert len(molecule_list) == 124327


def test_molecule_generation_string_1():
    """
    Test molecule string representation generation for two-bead
    molecules with three different bead types.
    """
    expected_strings = [
        "C1,",
        "C2,",
        "C3,",
        "C1 C1,0-1",
        "C1 C2,0-1",
        "C1 C3,0-1",
        "C2 C2,0-1",
        "C2 C3,0-1",
        "C3 C3,0-1",
    ]
    molecule_generator = MoleculeGenerator(2, ["C1", "C2", "C3"])
    for i, molecule_repr in enumerate(
        molecule_generator.generate_string_representations()
    ):
        assert molecule_repr == expected_strings[i]


def test_molecule_generation_string_2():
    """
    Test molecule string representation generation for two-bead
    molecules with three different bead types.
    """
    expected_strings = [
        "K0,",
        "K1,",
        "K2,",
        "K0 K0,0-1",
        "K0 K1,0-1",
        "K0 K2,0-1",
        "K1 K1,0-1",
        "K1 K2,0-1",
        "K2 K2,0-1",
    ]
    molecule_generator = MoleculeGenerator(2, bead_types="K", number_of_bead_types=3)
    for i, molecule_repr in enumerate(
        molecule_generator.generate_string_representations()
    ):
        assert molecule_repr == expected_strings[i]


def test_molecule_generation_networkx_graphs():
    """
    Test molecule networkx generation for two-bead
    molecules with three different bead types.
    """
    expected_graphs = [
        (["K0"], []),
        (["K1"], []),
        (["K2"], []),
        (["K0", "K0"], [(0, 1)]),
        (["K0", "K1"], [(0, 1)]),
        (["K0", "K2"], [(0, 1)]),
        (["K1", "K1"], [(0, 1)]),
        (["K1", "K2"], [(0, 1)]),
        (["K2", "K2"], [(0, 1)]),
    ]
    molecule_generator = MoleculeGenerator(2, bead_types="K", number_of_bead_types=3)
    for i, graph in enumerate(molecule_generator.generate_networkx_graphs()):
        node_names = [node[1]["name"] for node in graph.nodes(data=True)]
        assert node_names == expected_graphs[i][0]
        assert list(graph.edges) == expected_graphs[i][1]


def test_molecule_generation_torch_graphs_1():
    """
    Test molecule torch graph generation for two-bead
    molecules with three different bead types.
    """
    expected_graphs = [
        ([[0]], []),
        ([[1]], []),
        ([[2]], []),
        ([[0], [0]], [[0, 1], [1, 0]]),
        ([[0], [1]], [[0, 1], [1, 0]]),
        ([[0], [2]], [[0, 1], [1, 0]]),
        ([[1], [1]], [[0, 1], [1, 0]]),
        ([[1], [2]], [[0, 1], [1, 0]]),
        ([[2], [2]], [[0, 1], [1, 0]]),
    ]
    molecule_generator = MoleculeGenerator(2, bead_types="K", number_of_bead_types=3)
    generator = molecule_generator.generate_torch_graphs(add_self_loops=False)
    for i, graph in enumerate(generator):
        assert (graph.x == torch.Tensor(expected_graphs[i][0])).all()
        assert (graph.edge_index == torch.Tensor(expected_graphs[i][1])).all()


def test_molecule_generation_torch_graphs_2():
    """
    Test molecule torch graph generation for two-bead
    molecules with three different bead types.
    """
    expected_graphs = [
        ([[0]], [[0], [0]]),
        ([[1]], [[0], [0]]),
        ([[2]], [[0], [0]]),
        ([[0], [0]], [[0, 1, 0, 1], [1, 0, 0, 1]]),
        ([[0], [1]], [[0, 1, 0, 1], [1, 0, 0, 1]]),
        ([[0], [2]], [[0, 1, 0, 1], [1, 0, 0, 1]]),
        ([[1], [1]], [[0, 1, 0, 1], [1, 0, 0, 1]]),
        ([[1], [2]], [[0, 1, 0, 1], [1, 0, 0, 1]]),
        ([[2], [2]], [[0, 1, 0, 1], [1, 0, 0, 1]]),
    ]
    molecule_generator = MoleculeGenerator(2, bead_types="K", number_of_bead_types=3)
    for i, graph in enumerate(
        molecule_generator.generate_torch_graphs(add_self_loops=True)
    ):
        assert (graph.x == torch.Tensor(expected_graphs[i][0])).all()
        assert (graph.edge_index == torch.Tensor(expected_graphs[i][1])).all()
