"""Tests for the C-extension bond generator function."""

# pylint: disable=protected-access, c-extension-no-member
from chespex.molecules import bond_generator
from chespex.molecules.bonds import test_bondsmodule


def test_bond_generator_1():
    """Test the bond generator function with a simple case."""
    beads = [0, 0, 0]
    result = bond_generator._generate(*beads)
    expected_size = test_bondsmodule.generate(beads)
    print(result)
    print(expected_size)
    assert len(result) == expected_size


def test_bond_generator_2():
    """Test the bond generator function with a more complex case."""
    beads = [0, 0, 1, 2, 3]
    result = bond_generator._generate(*beads)
    expected_size = test_bondsmodule.generate(beads)
    print(result)
    print(expected_size)
    assert len(result) == expected_size


def test_bond_generator_3_slow():
    """Test the bond generator function with another case."""
    beads = [0, 0, 1, 1, 2, 3]
    result = bond_generator._generate(*beads)
    expected_size = test_bondsmodule.generate(beads)
    print(result)
    print(expected_size)
    assert len(result) == expected_size
