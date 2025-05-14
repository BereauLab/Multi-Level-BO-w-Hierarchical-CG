"""Module containing a simple data frame class."""

from typing import Any
import json
import numpy as np


class Dataframe:
    """
    A simple data frame class.
    In comparison to pandas dataframes, the data is stored in a list of dictionaries.
    This makes it easier to add entries to the dataframe one by one.
    """

    def __init__(self, columns: None | list[str] = None) -> None:
        self.data = []
        self.columns = columns

    def __convert_to_dict(self, entry: dict | list | tuple) -> bool:
        if isinstance(entry, (list, tuple)):
            if self.columns is None:
                raise ValueError(
                    "Column names must be defined to add entries as list or tuple."
                )
            if len(entry) != len(self.columns):
                raise ValueError(
                    "Length of entry does not match the number of columns."
                )
            entry = dict(zip(self.columns, entry))
        if isinstance(entry, dict):
            if self.columns is not None:
                for column in self.columns:
                    if column not in entry:
                        raise ValueError(f"Column {column} is missing in the entry.")
            else:
                self.columns = list(entry.keys())
        return entry

    def append(
        self,
        entry: list | dict,
    ) -> None:
        """
        Appends an entry to the dataframe.
        :param entry: Entry to append
        """
        entry = self.__convert_to_dict(entry)
        self.data.append(entry)

    def __setitem__(self, index: int | str, entry: dict | list | tuple) -> None:
        """
        Setter function for the dataframe.
        If the index is an integer, the entry is set as a row.
        If the index is a string, the entry is set as a column.
        :param index: Index of the entry
        :param entry: Entry to set
        """
        if isinstance(index, int):
            if index < 0 or index >= len(self.data):
                raise IndexError("Index out of range.")
            entry = self.__convert_to_dict(entry)
            self.data[index] = entry
        elif isinstance(index, str):
            if index not in self.columns:
                raise ValueError(f"Column {index} does not exist.")
            if isinstance(entry, dict):
                raise ValueError("Cannot set a column to a dictionary.")
            if isinstance(entry, list) or isinstance(entry, tuple):
                if len(entry) != len(self.data):
                    raise ValueError(
                        "Length of entry does not match the number of rows."
                    )
                for i, row in enumerate(self.data):
                    row[index] = entry[i]
            else:
                raise ValueError("Entry must be a list or tuple.")
        else:
            raise ValueError("Index must be an integer or a string.")

    def __iter__(self):
        """
        Returns an iterator over the rows of the dataframe.
        """
        return iter(self.data)

    def __getitem__(self, index: int | slice | str | tuple[str, str, Any]) -> dict:
        """
        Getter function for the dataframe.
        If the index is an integer, the row is returned.
        If the index is a string, the column is returned (as a numpy array).
        If the index is a tuple of two strings and an additional value, the column is returned
            for the rows where the second column matches the given value.
        If a column is returned, the result is converted to a numpy array.
        :param index: Index of the entry
        :return: The entry at the given index
        """
        if isinstance(index, int):
            return self.data[index]
        if isinstance(index, str):
            if index not in self.columns:
                raise ValueError(f"Column {index} does not exist.")
            if isinstance(self.data[0][index], (int, float, complex)):
                return np.array([row[index] for row in self.data])
            return [row[index] for row in self.data]
        if isinstance(index, slice):
            dataframe = Dataframe(columns=self.columns)
            dataframe.data = self.data[index]
            return dataframe
        if (
            isinstance(index, tuple)
            and isinstance(index[0], str)
            and isinstance(index[1], str)
        ):
            if index[0] not in self.columns:
                raise ValueError(f"Column {index[0]} does not exist.")
            if index[1] not in self.columns:
                raise ValueError(f"Column {index[1]} does not exist.")
            result = [row[index[0]] for row in self.data if row[index[1]] == index[2]]
            if isinstance(self.data[0][index[0]], (int, float, complex)):
                return np.array(result)
            return result
        raise ValueError("Index must be an integer or a string.")

    def __len__(self) -> int:
        """
        Returns the length of the dataframe.
        """
        return len(self.data)

    def save(self, path: str, **kwargs) -> None:
        """
        Saves the dataframe to a json file.
        :param path: Path to the json file
        :param kwargs: Additional keyword arguments for the json.dump function
        """
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.data, file, **kwargs)

    def __repr__(self) -> str:
        return f"Dataframe(columns={self.columns})"

    def filter(self, column: str, value: Any) -> "Dataframe":
        """
        Filters the dataframe by a column value.
        :param column: Name of the column to filter
        :param value: Value to filter by
        :return: The filtered dataframe
        """
        dataframe = Dataframe(columns=self.columns)
        for row in self.data:
            if row[column] == value:
                dataframe.append(row)
        return dataframe

    def sort(self, column: str, reverse: bool = False) -> "Dataframe":
        """
        Sorts the dataframe by a column.
        :param column: Name of the column to sort by
        :param reverse: If True, the dataframe is sorted in reverse order
        :return: The sorted dataframe
        """
        dataframe = Dataframe(columns=self.columns)
        for row in sorted(self.data, key=lambda x: x[column], reverse=reverse):
            dataframe.append(row)
        return dataframe

    @staticmethod
    def load(path: str, **kwargs) -> "Dataframe":
        """
        Loads a dataframe from a json file.
        :param path: Path to the json file
        :param kwargs: Additional keyword arguments for the json.load function
        :return: The loaded dataframe
        """
        # Load the data
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file, **kwargs)
        # Create the dataframe
        dataframe = Dataframe()
        dataframe.data = data
        dataframe.columns = list(data[0].keys())
        # Check if all rows have the same columns
        for row in data:
            if (
                len(set(row.keys()) - set(dataframe.columns)) > 0
                or len(set(dataframe.columns) - set(row.keys())) > 0
            ):
                raise ValueError("All rows must have the same columns.")
        return dataframe
