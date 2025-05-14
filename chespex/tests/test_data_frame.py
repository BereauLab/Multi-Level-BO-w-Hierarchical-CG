"""Tests for the data frame class that holds the data for the Bayesian optimization."""

import numpy as np
from chespex.optimization.data_frame import Dataframe


def test_add_data():
    """
    Test adding data to the data frame.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    assert data_frame.data == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_add_data_dict():
    """
    Test adding data to the data frame as a dictionary.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append({"a": 1, "b": 2})
    data_frame.append({"a": 3, "b": 4})
    assert data_frame.data == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]


def test_add_data_dict_missing_column():
    """
    Test adding data to the data frame as a dictionary with a missing column.
    """
    data_frame = Dataframe(columns=["a", "b"])
    try:
        data_frame.append({"a": 1})
    except ValueError as e:
        assert str(e) == "Column b is missing in the entry."


def test_set_row():
    """
    Test setting a row in the data frame.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    data_frame[0] = [5, 6]
    assert data_frame.data == [{"a": 5, "b": 6}, {"a": 3, "b": 4}]


def test_set_row_out_of_range():
    """
    Test setting a row in the data frame that is out of range.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    try:
        data_frame[2] = [5, 6]
    except IndexError as e:
        assert str(e) == "Index out of range."


def test_set_column():
    """
    Test setting a column in the data frame.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    data_frame["a"] = [5, 6]
    assert data_frame.data == [{"a": 5, "b": 2}, {"a": 6, "b": 4}]


def test_set_column_missing_column():
    """
    Test setting a column in the data frame that does not exist.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    try:
        data_frame["c"] = [5, 6]
    except ValueError as e:
        assert str(e) == "Column c does not exist."


def test_set_column_dict():
    """
    Test setting a column in the data frame with a dictionary.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    try:
        data_frame["a"] = {"a": 5, "b": 6}
    except ValueError as e:
        assert str(e) == "Cannot set a column to a dictionary."


def test_set_column_length_mismatch():
    """
    Test setting a column in the data frame with a length mismatch.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    try:
        data_frame["a"] = [5]
    except ValueError as e:
        assert str(e) == "Length of entry does not match the number of rows."


def test_get_row():
    """
    Test getting a row from the data frame.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    assert data_frame[0] == {"a": 1, "b": 2}


def test_get_column():
    """
    Test getting a column from the data frame.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    assert np.all(data_frame["a"] == np.array([1, 3]))


def test_get_column_with_filter():
    """
    Test getting a column from the data frame with a filter.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    data_frame.append([5, 6])
    assert np.all(data_frame["a", "b", 4] == np.array([3]))


def test_save_to_file(tmp_path):
    """
    Test saving the data frame to a file.
    """
    data_frame = Dataframe(columns=["a", "b"])
    data_frame.append([1, 2])
    data_frame.append([3, 4])
    path = tmp_path / "data.csv"
    data_frame.save(path)
    with open(path, "r", encoding="utf-8") as file:
        assert file.read() == '[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'


def test_load_from_file(tmp_path):
    """
    Test loading the data frame from a file.
    """
    data = '[{"a": 1, "b": 2}, {"a": 3, "b": 4}]'
    path = tmp_path / "data.csv"
    with open(path, "w", encoding="utf-8") as file:
        file.write(data)
    data_frame = Dataframe.load(path)
    assert data_frame.data == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
