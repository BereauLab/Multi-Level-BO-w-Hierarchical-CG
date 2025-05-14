"""Simulation class for GROMACS simulations."""

from typing import Callable, Optional
import os
import shutil
import subprocess
from time import sleep
import numpy as np
import pandas as pd
from .helper import add_simulation_task_to_queue


class Simulation:
    """
    Class for running GROMACS simulations using python.
    """

    def __init__(self, *input_files: str, **config: bool):
        """
        Initialize a simulation object with input files and configuration parameters.
        :param input_files: List of input files for the simulation. Required files are
            'tpr', 'top', 'gro', and 'mdp'. Optional files are 'ndx', 'res', and 'log'.
            The filename of the 'tpr' file is used for all other simulation output files.
            If a 'log' file is provided, the standard output of the simulation is written to it.
            If a 'res' file is provided, it is used for restraining the system.
        :param config: Configuration parameters for the simulation. The following keys are
            supported:
            - 'restrain': Boolean value indicating whether to restrain the system during the
                simulation using the coordinates from the 'gro' file. If a 'res' file is provided,
                it is used for restraining the system independently of this parameter. By default,
                the system is not restrained.
            - 'maxwarn': Integer value for the maximum number of warnings allowed during the
                simulation. By default, the maximum number of warnings is set to 1.
        """
        self.input_files = {
            filename.split(".")[-1]: filename
            for filename in input_files
            if filename is not None
        }
        if (
            len(
                [
                    e
                    for e in ["tpr", "top", "gro", "mdp"]
                    if e in self.input_files.keys()
                ]
            )
            != 4
        ):
            raise ValueError(
                "Not all required input files are provided, found only the following extensionsions:"
                + f"{list(self.input_files.keys())}"
            )
        self.config = config
        self.prepared = False
        self.out_coord = os.path.splitext(self.input_files["tpr"])[0] + ".gro"

    def modify_mdp(self, key: str | dict, value: str = None):
        """
        Modify the value of a key in the MDP file.
        :param key: Key in the MDP file to modify. Alternatively, a dictionary of key-value
            pairs can be provided to modify multiple keys at once.
        :param value: New value for the key.
        """
        if isinstance(key, str):
            if value is None:
                raise ValueError("Value must be provided if key is a string")
            key = {key: value}
        with open(self.input_files["mdp"], "r", encoding="utf-8") as f:
            lines = f.readlines()
        with open(self.input_files["mdp"], "w", encoding="utf-8") as f:
            for line in lines:
                found = False
                for k, v in key.items():
                    if line.startswith(k):
                        f.write(f'{line.split("=")[0]}= {v}\n')
                        found = True
                        break
                if not found:
                    f.write(line)

    def delete_outputs(self, *file_extensions: str):
        """
        Delete output files of the simulation with specific file extensions.
        :param file_extensions: List of file extensions to delete.
        """
        filebase = os.path.splitext(self.input_files["tpr"])[0]
        for extension in file_extensions:
            extension = extension[1:] if extension[0] == "." else extension
            filename = f"{filebase}.{extension}"
            if os.path.exists(filename):
                os.remove(filename)

    def prepare(self):
        """
        Prepare the GROMACS simulation by creating the TPR file.
        """
        maxwarn = self.config.get("maxwarn", 1)
        command = (
            f'gmx grompp -f {self.input_files["mdp"]} -c {self.input_files["gro"]} '
            + f'-p {self.input_files["top"]} -o {self.input_files["tpr"]} '
            + f'-po {self.input_files["tpr"].replace("tpr", "out.mdp")} -maxwarn {maxwarn}'
        )
        if "ndx" in self.input_files.keys():
            command += f' -n {self.input_files["ndx"]}'
        if "res" in self.input_files.keys():
            command += f' -r {self.input_files["res"]}'
        elif self.config.get("restrain", False):
            command += f' -r {self.input_files["gro"]}'
        if "log" in self.input_files.keys() and ">>" not in command:
            command += f' >> {self.input_files["log"]} 2>&1'
        subprocess.run(command, shell=True, check=True)
        self.prepared = True

    def submit_to_queue(self):
        """
        Submit the simulation to the queue.
        """
        if not self.prepared:
            self.prepare()
        filename = os.path.splitext(self.input_files["tpr"])[0]
        add_simulation_task_to_queue(filename)

    def run(self, extra_args: str = ""):
        """
        Run the simulation locally.
        """
        if not self.prepared:
            self.prepare()
        filename = os.path.splitext(self.input_files["tpr"])[0]
        command = f"gmx mdrun -deffnm {filename} {extra_args}"
        if "log" in self.input_files.keys() and ">>" not in command:
            command += f' >> {self.input_files["log"]} 2>&1'
        subprocess.run(command, shell=True, check=True)

    def wait_for_file(self, filename: str, waiting_timesteps: float = 0.3):
        """
        Wait for a file to be created.
        :param filename: Name of the file to wait for.
        :param waiting_timesteps: Time to wait between checks for the file.
        """
        while not os.path.exists(filename):
            sleep(waiting_timesteps)

    def submit_and_wait(self):
        """
        Prepare the simulation, submit it to the queue, and wait for the output files.
        This is a convenience method that calls the 'prepare', 'submit_to_queue', and
        'wait_for_file' methods. I.e. the simulation is submitted to the queue and the
        method waits for the simulation to finish.
        """
        self.submit_to_queue()
        filename = os.path.splitext(self.input_files["tpr"])[0] + ".gro"
        self.wait_for_file(filename)


class SimulationSetup:
    """
    Class for setting up GROMACS simulations using python.
    This class helps with combining topologies and coordinate files.
    """

    def __init__(
        self,
        input_files: Optional[str] = None,
        output_files: Optional[str] = None,
        *,
        in_topology: Optional[str] = None,
        in_coordinates: Optional[str] = None,
        out_topology: Optional[str] = None,
        out_coordinates: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        """
        Initialize a simulation setup object with input and output files.
        :param input_files: Name of the input files without file extension.
        :param output_files: Name of the output files without file extension.
        :param in_topology: Name of the input topology file. Default is '<input_files>.top'.
        :param in_coordinates: Name of the input coordinate file. Default is '<input_files>.gro'.
        :param out_topology: Name of the output topology file. Default is '<output_files>.top'.
        :param out_coordinates: Name of the output coordinate file. Default is '<output_files>.gro'.
        :param log_file: Name of the log file for the simulation setup. If not provided, the
            standard output of the gmx simulation setup is written to the console.
        """
        if input_files is None and (in_topology is None or in_coordinates is None):
            raise ValueError(
                "Either 'input_files' or 'in_topology' and 'in_coordinates' must be provided"
            )
        if (
            input_files is None
            and output_files is None
            and (out_topology is None or out_coordinates is None)
        ):
            raise ValueError(
                "Either 'output_files' or 'out_topology' and 'out_coordinates' must be provided"
            )
        if output_files is None:
            output_files = input_files
        self.in_top = in_topology if in_topology is not None else f"{input_files}.top"
        self.in_coord = (
            in_coordinates if in_coordinates is not None else f"{input_files}.gro"
        )
        self.out_top = (
            out_topology if out_topology is not None else f"{output_files}.top"
        )
        self.out_coord = (
            out_coordinates if out_coordinates is not None else f"{output_files}.gro"
        )
        self.out_index = None
        if self.in_top != self.out_top:
            shutil.copyfile(self.in_top, self.out_top)
        if self.in_coord != self.out_coord:
            shutil.copyfile(self.in_coord, self.out_coord)
        self.log_file = log_file

    def add_charge(
        self,
        charge_to_add: int,
        position_info: list[float | tuple[float, float]],
        d: float = 1.0,
    ):
        """
        Add a charge to the coordinate file. If the charge is negative, chloride ions are added. If
        the charge is positive, sodium ions are added.
        :param charge_to_add: Charge to add to the system.
        :param position_info: List of floats or float-tuples which specify the position (and its
            allowed deviation) of the charges in the coordinate file. The format of the tuples
            is (x, dx), where x is the position of the charge and dx is the allowed deviation
            from the position. If only a single float is provided, the deviation is set to 'd'.
            The list should contain three tuples for the x, y, and z coordinates.
        :param d: Deviation for the position of the charges. If a float is provided, the deviation
            is set to the same value for all coordinates if not specified in 'position_info'.
        """
        charge_type = "NA" if charge_to_add > 0 else "CL"
        net_charge = abs(charge_to_add)
        position_info = [p if isinstance(p, tuple) else (p, d) for p in position_info]
        # Determine the positions of the charges
        positions = []
        for _ in range(net_charge):
            position = [
                p[0] + p[1] * (2 * (0.5 - np.random.rand())) for p in position_info
            ]
            positions.append(position)
        # Add charges to the coordinate file
        with open(self.out_coord, "r+", encoding="utf-8") as coord_file:
            lines = coord_file.readlines()
            lines[1] = f"{int(lines[1]) + net_charge}\n"
            box = lines.pop()
            residue_index = int(lines[-1][:5]) + 1
            atom_index = int(lines[-1][15:20]) + 1
            for i, position in enumerate(positions):
                lines.append(
                    f"{residue_index + i:>5}{charge_type:<5}"
                    + f"{charge_type:>5}{atom_index + i:>5}"
                    + f"{position[0]:>8.3f}{position[1]:>8.3f}{position[2]:>8.3f}\n"
                )
            lines.append(box)
            coord_file.seek(0)
            coord_file.writelines(lines)
        # Add charges to the topology file
        with open(self.out_top, "a", encoding="utf-8") as top_file:
            top_file.write(f"{charge_type:<8} {net_charge:>9}\n")

    def add_molecule(
        self,
        input_file: str,
        positions: list[tuple[float, float, float]],
        deviations: tuple[float, float, float] = (1.0, 1.0, 1.0),
        coordinate_file: str = None,
        topology_file: str = None,
        copy_topology: bool = True,
        radius: float = 0.15,
        ntry: int = 50000,
    ):
        """
        Add a molecule to the coordinate and topology files.
        :param input_file: Name of the coordinate and topology (.itp) file for the molecule without
            file extension.
        :param positions: List of tuples which specify the positions of the molecules to insert
            into the coordinate file. The format of the tuples is (x, y, z), where x, y, and z are
            the positions of the molecules. The number of entries in the list specifies the number
            of molecules to insert.
        :param deviation: Tuple of floats which specify the allowed deviation from the positions
            of the molecules. The format of the tuple is (dx, dy, dz), where dx, dy, and dz are the
            allowed deviations from the positions given in 'position_info'. If only a single float
            is provided, the deviation is set to the same value for all coordinates.
        :param coordinate_file: Name of the coordinate file for the molecule. Default is
            '<input_file>.gro'.
        :param topology_file: Name of the topology file for the molecule. Default is
            '<input_file>.top'.
        :param copy_topology: Boolean value indicating whether to copy the topology file into the
            output topology file. If set to 'False', the topology file is included in the output
            topology file using an '#include' statement.
        :param radius: Radius of the beads of the molecule to insert into the coordinate file. Used
            for the 'gmx insert-molecules' command.
        :param ntry: Number of attempts to insert the molecule into the coordinate file. Used for
            the 'gmx insert-molecules' command.
        """
        # Prepare input coordinates and topology files
        coordinate_file = (
            f"{input_file}.gro" if coordinate_file is None else coordinate_file
        )
        topology_file = f"{input_file}.itp" if topology_file is None else topology_file
        # Add molecule to the coordinate file
        deviations = deviations if isinstance(deviations, tuple) else (deviations,) * 3
        position_filename = f"{self.out_coord}.tmp.dat"
        for position in positions:
            with open(position_filename, "w", encoding="utf-8") as position_file:
                position_file.write(
                    f"{position[0]:.4f} {position[1]:.4f} {position[2]:.4f}\n"
                )
            command = (
                f"gmx insert-molecules -f {self.out_coord} -ci {coordinate_file} "
                + f"-o {self.out_coord} -ip {position_filename} "
                + f"-dr {deviations[0]:.1f} {deviations[1]:.1f} {deviations[2]:.1f} "
                + f"-rot xyz -nmol 1 -try {ntry} -radius {radius:.2f}"
            )
            if self.log_file is not None:
                command += f" >> {self.log_file} 2>&1"
            subprocess.run(command, shell=True, check=True)
            backup_filename = os.path.join(
                f"{os.path.dirname(self.out_coord)}",
                f"#{os.path.basename(self.out_coord)}.1#",
            )
            if os.path.exists(backup_filename):
                os.remove(backup_filename)
        os.remove(position_filename)
        # Get molecule name from molecule topology (needed for system topology file)
        with open(topology_file, "r", encoding="utf-8") as ligand_top_file:
            ligand_lines = ligand_top_file.readlines()
        ligand_name_index = [
            i for i, line in enumerate(ligand_lines) if "moleculetype" in line
        ][0]
        ligand_name_index += 1
        while ligand_lines[ligand_name_index][0] == ";" and ligand_name_index < len(
            ligand_lines
        ):
            ligand_name_index += 1
        ligand_name = ligand_lines[ligand_name_index].split()[0]
        # Add molecule to the system topology file
        with open(self.out_top, "r+", encoding="utf-8") as top_file:
            lines = top_file.readlines()
            insert_index = [i for i, line in enumerate(lines) if "system" in line][
                0
            ] - 1
            if copy_topology:
                if len(lines[insert_index - 1]) > 1:
                    ligand_lines.insert(0, "\n")
                lines[insert_index:insert_index] = ligand_lines
            else:
                topology_file = os.path.relpath(
                    topology_file, os.path.dirname(self.out_top)
                )
                insert_text = f'#include "{topology_file}"\n'
                if len(lines[insert_index - 1]) > 1:
                    insert_text = "\n" + insert_text
                lines.insert(insert_index, insert_text)
            lines.append(f"{ligand_name:<8} {len(positions):>9}\n")
            top_file.seek(0)
            top_file.writelines(lines)

    def create_index(
        self,
        index_filename: str,
        selections: dict[
            str, Callable[[pd.DataFrame], pd.DataFrame | list[pd.DataFrame]]
        ],
    ):
        """
        Create an index file for the system coordinate file.
        Similar to the 'make_ndx' command in GROMACS, but programmable with python.
        :param index_filename: Name of the index file to create.
        :param selections: Dictionary of selections to apply to the coordinate file. The keys are
            the names of the selections and the values are functions that take a pandas DataFrame
            as input and return a pandas DataFrame or a list / groupby object of pandas DataFrames.
            In the latter case, an 1-based index is added to the selection name. The column names
            of the DataFrame are 'resid', 'resname', 'atomname', 'atomid', 'x', 'y',
        """
        # Parse coordinate file
        with open(self.out_coord, "r", encoding="utf-8") as coord_file:
            lines = [l for l in coord_file.readlines() if len(l.strip()) > 0]
        data = []
        for line in lines[2:-1]:
            data.append(
                [
                    int(line[:5]),
                    line[5:10].strip(),
                    line[10:15].strip(),
                    int(line[15:20]),
                    float(line[20:28]),
                    float(line[28:36]),
                    float(line[36:44]),
                ]
            )
        data = pd.DataFrame(
            data, columns=["resid", "resname", "atomname", "atomid", "x", "y", "z"]
        )
        # Apply selections to data and add to dict. If the selection returns a groupby object
        # or list, iterate over the groups and add them to the dict with an index.
        selection_data = {}
        for selection_name, selection in selections.items():
            selection_group = selection(data)
            if isinstance(selection_group, pd.core.groupby.generic.DataFrameGroupBy):
                for i, (_, sg) in enumerate(selection_group):
                    selection_data[f"{selection_name}{i+1}"] = sg
            elif isinstance(selection_group, list):
                for i, sg in enumerate(selection_group):
                    selection_data[f"{selection_name}{i+1}"] = sg
            else:
                selection_data[selection_name] = selection_group
        # Write selections to index file
        with open(index_filename, "w", encoding="utf-8") as index_file:
            for selection_name, selection_group in selection_data.items():
                atom_indices = selection_group["atomid"].values
                index_file.write(f"[ {selection_name} ]\n")
                for i, atom_index in enumerate(atom_indices):
                    index_file.write(f"{atom_index:>4}")
                    if i % 15 == 14 or i == len(atom_indices) - 1:
                        index_file.write("\n")
                    else:
                        index_file.write(" ")
        # Set index filename as object attribute
        self.out_index = index_filename

    def replace_string_in_topology(self, search: str, replacement: str):
        """
        Replace a string in the topology file.
        :param search: String to search for in the topology file.
        :param replacement: String to replace the search string with.
        """
        with open(self.out_top, "r", encoding="utf-8") as top_file:
            lines = top_file.read()
        if isinstance(replacement, list):
            replacement = "\n".join(replacement)
        lines = lines.replace(search, replacement)
        with open(self.out_top, "w", encoding="utf-8") as top_file:
            top_file.write(lines)
