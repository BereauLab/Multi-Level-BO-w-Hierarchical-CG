from typing import Optional, Any
import os
import logging
import json
import warnings
from time import sleep
from subprocess import check_output
from itertools import product
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

logging.getLogger("pymbar").setLevel(logging.ERROR)

from alchemlyb.estimators import MBAR
from alchemlyb.parsing.gmx import extract_u_nk
from chespex.simulation import Simulation, SimulationSetup
from chespex.molecules import Molecule

SIMULATE_ALL_AT_ONCE = True


def run_simulation_locally(
    directory: str,
    topology_file: str,
    coordinate_file: str,
    simulation_name: str,
    mdp_file: str,
    log_file: Optional[str] = None,
    mdp_modifications: Optional[dict[str, str | float]] = None,
    delete_extensions: Optional[list[str]] = None,
):
    """
    Run a simulation locally.
    :param directory: The directory where the simulation files are stored.
    :param topology_file: The topology file for the simulation.
    :param coordinate_file: The coordinate file for the simulation.
    :param simulation_name: The name of the simulation.
    :param mdp_file: The MDP file for the simulation.
    :param log_file: The log file to write the output to. If not specified, the output
        is written to the standard output.
    :param mdp_modifications: Modifications to the MDP file.
    :param delete_extensions: Extensions of files to delete after the simulation.
    """
    simulation = Simulation(
        topology_file,
        coordinate_file,
        mdp_file,
        f"{directory}/{simulation_name}.tpr",
        f"{directory}/system.ndx",
        log_file,
    )
    if not os.path.exists(simulation.out_coord):
        if mdp_modifications is not None:
            simulation.modify_mdp(mdp_modifications)
        # minimization.submit_and_wait()
        simulation.run("-nt 8 -pin on")
        if delete_extensions is not None:
            simulation.delete_outputs(*delete_extensions)
    return simulation


def run_system_simulations(
    system: str,
    directory: str,
    molecule: Molecule,
    main_directory: str,
    total_charge: int,
    positions: list[tuple[float, float, float]],
    index_groups: dict[str, callable],
    minimization_mdp: str,
    equilibration_mdp: str,
    production_mdp: str,
    lambda_steps: int,
    log_file: Optional[str] = None,
    minimization_modifications: Optional[dict[str, str | float]] = None,
    equilibration_modifications: Optional[dict[str, str | float]] = None,
    production_modifications: Optional[dict[str, str | float]] = None,
):
    """
    Run free energy simulations for a molecule in a system.
    :param system: The name of the system.
    :param directory: The directory where the simulation files are stored.
    :param molecule: The molecule to simulate.
    :param main_directory: The main directory where the simulation files are stored.
    :param total_charge: The total charge of the molecule.
    :param positions: The positions to place the molecule in the system.
    :param index_groups: The index groups to create for the system.
    :param minimization_mdp: The MDP file for the minimization.
    :param equilibration_mdp: The MDP file for the equilibration.
    :param production_mdp: The MDP file for the production.
    :param lambda_steps: The number of lambda steps.
    :param log_file: The log file to write the output to. If not specified, the output
        is written to the standard output.
    :param minimization_modifications: Modifications to the minimization MDP file.
    :param equilibration_modifications: Modifications to the equilibration MDP file.
    :param production_modifications: Modifications to the production MDP file.
    """
    # Create directory
    directory = f"{main_directory}/{directory}"
    os.makedirs(directory, exist_ok=True)
    # Create simulation setup
    topology_file = f"{directory}/system.top"
    if (
        not os.path.exists(f"{directory}/system.gro")
        or not os.path.exists(f"{directory}/system.top")
        or not os.path.exists(f"{directory}/Lig.gro")
        or not os.path.exists(f"{directory}/Lig.itp")
    ):
        setup = SimulationSetup(
            f"input-files/{system}", f"{directory}/system", log_file=log_file
        )
        setup.add_charge(-total_charge, [0, 0, 0], d=1)
        molecule.write_simulation_files(f"{directory}/Lig", nrexcl=1)
        setup.add_molecule(
            f"{directory}/Lig",
            positions=positions,
            deviations=(2, 2, 1),
            copy_topology=False,
        )
        setup.create_index(f"{directory}/system.ndx", index_groups)
        topology_file = setup.out_top
    molecule_name = "".join([bead["name"] for bead in molecule.beads])
    minimization = run_simulation_locally(
        directory,
        topology_file,
        f"{directory}/system.gro",
        "minimization",
        minimization_mdp,
        log_file,
        minimization_modifications,
        ["trr", "edr"],
    )
    equilibration = run_simulation_locally(
        directory,
        topology_file,
        minimization.out_coord,
        "equilibration",
        equilibration_mdp,
        log_file,
        equilibration_modifications,
        ["trr", "edr"],
    )
    # Run production
    for lambd in range(lambda_steps):
        lambda_dir = f"{directory}/lambda-{lambd:02d}"
        os.makedirs(lambda_dir, exist_ok=True)
        production = Simulation(
            topology_file,
            equilibration.out_coord,
            production_mdp,
            f"{lambda_dir}/production.tpr",
            f"{directory}/system.ndx",
            log_file,
        )
        if not os.path.exists(production.out_coord) and not os.path.exists(
            production.input_files["tpr"]
        ):
            if production_modifications is None:
                production_modifications = {}
            production_modifications["init-lambda-state"] = lambd
            production_modifications["couple-moltype"] = molecule_name
            production.modify_mdp(production_modifications)
            production.submit_to_queue()


def calculate_free_energy(system_directory: str, lambda_steps: int) -> float:
    """
    Use MBAR to calculate the free energy difference for simulations
    stored in the specified directory.
    :param system_directory: The directory where the simulation files are stored.
    :param lambda_steps: The number of lambda steps. This is used to determine
        if the simulations are finished.
    """
    if not os.path.exists(f"{system_directory}/mbar.json"):
        xvg_files = []
        if os.path.islink(system_directory) and not os.path.exists(system_directory):
            # Symlink not accessible
            return
        for lambd in range(lambda_steps):
            if os.path.exists(f"{system_directory}/lambda-{lambd:02d}/production.xvg"):
                xvg_files.append(
                    f"{system_directory}/lambda-{lambd:02d}/production.xvg"
                )
        if len(xvg_files) == lambda_steps:
            print(f"{datetime.now()} Calculating MBAR for {system_directory}")
            u_nk_list = [extract_u_nk(f, T=305) for f in xvg_files]
            u_nk_combined = pd.concat(u_nk_list)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mbar = MBAR().fit(u_nk_combined)
            # Save the results
            free_energy = float(mbar.delta_f_.iloc[0, -1])
            d_free_energy = float(mbar.d_delta_f_.iloc[0, -1])
            delta = mbar.delta_f_.to_numpy()[0, 1:] - mbar.delta_f_.to_numpy()[0, :-1]
            with open(f"{system_directory}/mbar.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "free_energy": free_energy * 0.5924,  # Convert to kcal/mol
                        "d_free_energy": d_free_energy * 0.5924,
                        "delta": list(delta * 0.5924),
                    },
                    f,
                )
            del mbar
        else:
            print(
                "Simulations not finished yet: "
                + f"{len(xvg_files)}/{lambda_steps} for {system_directory}"
            )


def get_free_energy(system_directory: str | Path) -> float:
    """
    Get the free energy from the MBAR calculation.
    :param system_directory: The directory where the simulation files are stored.
    :return: The free energy.
    """
    if not (Path(system_directory) / "mbar.json").exists():
        return None
    with open(f"{system_directory}/mbar.json", "r", encoding="utf-8") as f:
        free_energy_data = json.load(f)
    return free_energy_data["free_energy"]


def run_water_simulations(
    directory: str,
    molecule: Molecule,
    total_charge: int,
    is_charged: bool = False,
    log_file: Optional[str] = None,
):
    """
    Run simulations for the molecule in water and in the membrane.
    :param directory: The directory where the simulation files are stored.
    :param molecule: The molecule to simulate.
    :param total_charge: The total charge of the molecule.
    :param is_charged: Whether the molecule is charged.
    :param log_file: The log file to write the output to. If not specified, the output
        is written to the standard output.
    """
    run_system_simulations(
        "water",
        "water",
        molecule,
        directory,
        total_charge,
        positions=[(1.6, 1.6, 1.6)],
        index_groups={
            "Lig": lambda x: x[x["resname"] == "LIG"],
            "Other": lambda x: x[(x["resname"] != "LIG")],
            "System": lambda x: x,
        },
        minimization_mdp="input-files/minimization-water.mdp",
        equilibration_mdp="input-files/equilibration-water.mdp",
        production_mdp=(
            "input-files/production-water-charge.mdp"
            if is_charged
            else "input-files/production-water.mdp"
        ),
        lambda_steps=36 if is_charged else 26,
        log_file=log_file,
    )


def run_dipc_center_simulations(
    directory: str,
    molecule: Molecule,
    total_charge: int,
    is_charged: bool = False,
    log_file: Optional[str] = None,
):
    """
    Run simulations for the molecule in the DIPC membrane.
    :param directory: The directory where the simulation files are stored.
    :param molecule: The molecule to simulate.
    :param total_charge: The total charge of the molecule.
    :param is_charged: Whether the molecule is charged.
    :param log_file: The log file to write the output to. If not specified, the output
        is written to the standard output.
    """
    run_system_simulations(
        "DIPC",
        "DIPC",
        molecule,
        directory,
        total_charge,
        positions=[(3.0, 3.0, 5.0)],
        index_groups={
            "Lig": lambda x: x[x["resname"] == "LIG"],
            "Membrane": lambda x: x[x["resname"] == "DIPC"],
            "Other": lambda x: x[(x["resname"] != "LIG") & (x["resname"] != "DIPC")],
            "System": lambda x: x,
        },
        minimization_mdp="input-files/minimization-center.mdp",
        equilibration_mdp="input-files/equilibration-center.mdp",
        production_mdp=(
            "input-files/production-center-charge.mdp"
            if is_charged
            else "input-files/production-center.mdp"
        ),
        lambda_steps=36 if is_charged else 26,
        log_file=log_file,
        equilibration_modifications={"colvars-active": "no", "colvars-config": ""},
        production_modifications={
            "nsteps": 400000,
            "colvars-active": "no",
            "colvars-config": "",
        },
    )


def run_dipc_interface_simulations(
    directory: str,
    molecule: Molecule,
    total_charge: int,
    is_charged: bool = False,
    log_file: Optional[str] = None,
):
    """
    Run simulations for the molecule in the DIPC membrane.
    :param directory: The directory where the simulation files are stored.
    :param molecule: The molecule to simulate.
    :param total_charge: The total charge of the molecule.
    :param is_charged: Whether the molecule is charged.
    :param log_file: The log file to write the output to. If not specified, the output
        is written to the standard output.
    """
    run_system_simulations(
        "DIPC",
        "DIPC_interface",
        molecule,
        directory,
        total_charge,
        positions=[(3.0, 3.0, 6.6)],
        index_groups={
            "Lig": lambda x: x[x["resname"] == "LIG"],
            "Membrane": lambda x: x[x["resname"] == "DIPC"],
            "Other": lambda x: x[(x["resname"] != "LIG") & (x["resname"] != "DIPC")],
            "System": lambda x: x,
        },
        minimization_mdp="input-files/minimization-center.mdp",
        equilibration_mdp="input-files/equilibration-interface.mdp",
        production_mdp=(
            "input-files/production-interface-charge.mdp"
            if is_charged
            else "input-files/production-interface.mdp"
        ),
        lambda_steps=36 if is_charged else 26,
        log_file=log_file,
        equilibration_modifications={"colvars-active": "no", "colvars-config": ""},
        production_modifications={
            "nsteps": 900000,
            "colvars-active": "no",
            "colvars-config": "",
        },
    )


def run_mixed_simulations(
    directory: str,
    molecule: Molecule,
    total_charge: int,
    is_charged: bool = False,
    log_file: Optional[str] = None,
):
    """
    Run simulations for the molecule in the mixed membrane.
    :param directory: The directory where the simulation files are stored.
    :param molecule: The molecule to simulate.
    :param total_charge: The total charge of the molecule.
    :param is_charged: Whether the molecule is charged.
    :param log_file: The log file to write the output to. If not specified, the output
        is written to the standard output.
    """
    system_directory = f"{directory}/MIX"
    os.makedirs(system_directory, exist_ok=True)
    if not os.path.exists(f"{system_directory}/colvars.config"):
        with open("input-files/colvars.config", "r", encoding="utf-8") as f:
            colvars = f.read()
        colvars = colvars.replace("system.ndx", f"{system_directory}/system.ndx")
        with open(f"{system_directory}/colvars.config", "w", encoding="utf-8") as f:
            f.write(colvars)
    run_system_simulations(
        "MIX",
        "MIX",
        molecule,
        directory,
        total_charge,
        positions=[(3.0, 3.0, 5.0)],
        index_groups={
            "Lig": lambda x: x[x["resname"] == "LIG"],
            "Membrane": lambda x: x[
                (x["resname"] == "DPPC")
                | (x["resname"] == "DIPC")
                | (x["resname"] == "CHOL")
            ],
            "DPPC_C1A_top": lambda x: x[
                (x["resname"] == "DPPC") & (x["resid"] < 60) & (x["atomname"] == "C1A")
            ],
            "DIPC_C1A_top": lambda x: x[
                (x["resname"] == "DIPC") & (x["resid"] < 60) & (x["atomname"] == "C1A")
            ],
            "DPPC_C1A_bottom": lambda x: x[
                (x["resname"] == "DPPC") & (x["resid"] > 60) & (x["atomname"] == "C1A")
            ],
            "DIPC_C1A_bottom": lambda x: x[
                (x["resname"] == "DIPC") & (x["resid"] > 60) & (x["atomname"] == "C1A")
            ],
            "Other": lambda x: x[
                (x["resname"] != "LIG")
                & (x["resname"] != "DPPC")
                & (x["resname"] != "DIPC")
                & (x["resname"] != "CHOL")
            ],
            "System": lambda x: x,
        },
        minimization_mdp="input-files/minimization-center.mdp",
        equilibration_mdp="input-files/equilibration-center.mdp",
        production_mdp=(
            "input-files/production-center-charge.mdp"
            if is_charged
            else "input-files/production-center.mdp"
        ),
        lambda_steps=36 if is_charged else 26,
        log_file=log_file,
        equilibration_modifications={
            "colvars-active": "yes",
            "colvars-config": f"{system_directory}/colvars.config",
        },
        production_modifications={
            "nsteps": 900000,
            "colvars-active": "yes",
            "colvars-config": f"{system_directory}/colvars.config",
        },
    )


def run_mixed_interface_simulations(
    directory: str,
    molecule: Molecule,
    total_charge: int,
    is_charged: bool = False,
    log_file: Optional[str] = None,
):
    """
    Run simulations for the molecule in the mixed membrane.
    :param directory: The directory where the simulation files are stored.
    :param molecule: The molecule to simulate.
    :param total_charge: The total charge of the molecule.
    :param is_charged: Whether the molecule is charged.
    :param log_file: The log file to write the output to. If not specified, the output
        is written to the standard output.
    """
    system_directory = f"{directory}/MIX_interface"
    os.makedirs(system_directory, exist_ok=True)
    if not os.path.exists(f"{system_directory}/colvars.config"):
        with open("input-files/colvars.config", "r", encoding="utf-8") as f:
            colvars = f.read()
        colvars = colvars.replace("system.ndx", f"{system_directory}/system.ndx")
        with open(f"{system_directory}/colvars.config", "w", encoding="utf-8") as f:
            f.write(colvars)
    run_system_simulations(
        "MIX",
        "MIX_interface",
        molecule,
        directory,
        total_charge,
        positions=[(3.0, 3.0, 5.0)],
        index_groups={
            "Lig": lambda x: x[x["resname"] == "LIG"],
            "Membrane": lambda x: x[
                (x["resname"] == "DPPC")
                | (x["resname"] == "DIPC")
                | (x["resname"] == "CHOL")
            ],
            "DPPC_C1A_top": lambda x: x[
                (x["resname"] == "DPPC") & (x["resid"] < 60) & (x["atomname"] == "C1A")
            ],
            "DIPC_C1A_top": lambda x: x[
                (x["resname"] == "DIPC") & (x["resid"] < 60) & (x["atomname"] == "C1A")
            ],
            "DPPC_C1A_bottom": lambda x: x[
                (x["resname"] == "DPPC") & (x["resid"] > 60) & (x["atomname"] == "C1A")
            ],
            "DIPC_C1A_bottom": lambda x: x[
                (x["resname"] == "DIPC") & (x["resid"] > 60) & (x["atomname"] == "C1A")
            ],
            "Other": lambda x: x[
                (x["resname"] != "LIG")
                & (x["resname"] != "DPPC")
                & (x["resname"] != "DIPC")
                & (x["resname"] != "CHOL")
            ],
            "System": lambda x: x,
        },
        minimization_mdp="input-files/minimization-center.mdp",
        equilibration_mdp="input-files/equilibration-interface.mdp",
        production_mdp=(
            "input-files/production-interface-charge.mdp"
            if is_charged
            else "input-files/production-interface.mdp"
        ),
        lambda_steps=36 if is_charged else 26,
        log_file=log_file,
        equilibration_modifications={
            "colvars-active": "yes",
            "colvars-config": f"{system_directory}/colvars.config",
        },
        production_modifications={
            "nsteps": 1200000,
            "colvars-active": "yes",
            "colvars-config": f"{system_directory}/colvars.config",
        },
    )


def load_result_for_molecule(molecule_dir: Path, default: int = 0):
    """
    Load and combine mbar results for a molecule.
    :param molecule_dir: The directory where the simulation files are stored.
    :param default: The default value to return if the molecule preference is higher for
        the water phase.
    :return: The free energy difference for DIPC -> MIX. If files are not found, None is
        returned.
    """
    water_mbar = get_free_energy(molecule_dir / "water")
    mix_mbar = get_free_energy(molecule_dir / "MIX")
    if water_mbar is None or mix_mbar is None:
        return None
    if water_mbar > mix_mbar:
        return 0.5 + min(water_mbar - mix_mbar, 25) / (25 / 2)
    mix_interface_mbar = get_free_energy(molecule_dir / "MIX_interface")
    dipc_mbar = get_free_energy(molecule_dir / "DIPC")
    if mix_interface_mbar is None or dipc_mbar is None:
        return None
    if mix_interface_mbar > mix_mbar:
        return min(mix_interface_mbar - mix_mbar, 3) / (3 / 0.5)
    ddg = (water_mbar - dipc_mbar) - (water_mbar - mix_mbar)
    return min(ddg, default)


def load_completed_molecules_list(filepath: str, default_level: int = 0):
    """
    Load the list of completed molecules from the molecules.list file.
    """
    completed_molecules = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if line.count(";") < 1:
                path = Path(f"simulations/level-{default_level}/{line.strip()}")
                name = line.strip().replace("_", " ")
                completed_molecules.append({"path": path, "name": name, "level": 0})
            else:
                molecule_dirname, level = line.strip().split(";")[:2]
                path = Path(
                    f"simulations/level-{level.strip()}/{molecule_dirname.strip()}"
                )
                name = molecule_dirname.replace("_", " ")
                completed_molecules.append(
                    {"path": path, "name": name.strip(), "level": level.strip()}
                )
    return completed_molecules


def wait_for_simulations_finished(molecules: list[dict[str, Any]]):
    """
    Wait for all simulations to finish.
    :param molecules: The list of molecules to wait for. It should be a list of
        dictionaries with the key "path" and "name".
    """
    while True:
        completed = True
        for molecule in molecules:
            if load_result_for_molecule(molecule["path"]) is None:
                print(f"Waiting for {molecule['name']} to finish", end="\r", flush=True)
                completed = False
                break
        if completed:
            break
        sleep(3)


def wait_for_simulation_results(directory, number_of_files):
    """
    Wait for the number of "production.gro" files in the given directory and subdirectories
    to reach the given number.
    :param directory: The directory where the files are stored.
    :param number_of_files: The number of files that are expected.
    """
    warning_printed = False
    while True:
        command_tpr = f'find -L {directory} -name "*production.tpr" | wc -l'
        total_count = int(check_output(command_tpr, shell=True).decode().strip())
        command_gro = f'find -L {directory} -name "*production.gro" | wc -l'
        finished_count = int(check_output(command_gro, shell=True).decode().strip())
        if total_count != number_of_files and not warning_printed:
            print("Warning: Not all simulations are setup yet. We continue anyway.")
            warning_printed = True
        if finished_count >= total_count:
            break
        sleep(3)


def wait_for_mbar_results(*directories):
    """
    Wait for the mbar.json files to be created in the given directories.
    :param directories: The directories where the mbar.json files are expected.
    """
    while True:
        completed = True
        for directory in directories:
            if not os.path.exists(f"{directory}/mbar.json"):
                completed = False
                break
        if completed:
            break
        sleep(3)


def simulate_molecule(
    db, db_molecule, level, simulate_all_at_once=SIMULATE_ALL_AT_ONCE
):
    """
    Run the simulations for a molecule in the database.
    :param db: The database collection for the molecules.
    :param db_molecule: The molecule object from the database.
    :param level: The level of the molecule in the database.
    """
    print(f'Running calculations for {db_molecule["name"]} on level {level}')
    # We fetch the molecule from the database again to get all fields/attributes.
    db_molecule = db.find_one({"name": db_molecule["name"]})
    # Reconstruct molecule
    molecule = Molecule.reconstruct(
        db_molecule["bead_names"],
        db_molecule["node_features"],
        db_molecule["edge_index"],
        {"size": int, "class": int, "charge": int, "oco_w_tfe": float},
    )
    # Setup the simulation
    directory = f"simulations/level-{level}/{db_molecule['name'].replace(' ', '_')}"
    os.makedirs(directory, exist_ok=True)
    total_charge = sum([bead["charge"] - 1 for bead in molecule.beads])
    is_charged = sum([abs(bead["charge"] - 1) for bead in molecule.beads]) > 0
    # Replace bead types for level 0 and 1
    if level < 2:
        with open("../bead_types/mapping.json", "r", encoding="utf-8") as mapping_file:
            mapping = json.load(mapping_file)[level]
        mapping = {
            bt_name: f'{["K","L"][level]}{mi+1}'
            for mi, bt_name in enumerate(mapping.keys())
        }
        for bead in molecule.beads:
            for btmo, btmn in mapping.items():
                bead["name"] = bead["name"].replace(btmo, btmn)
    # Run the simulations
    steps = 36 if is_charged else 26
    if not os.path.exists(f"{directory}/water/mbar.json"):
        run_water_simulations(
            directory, molecule, total_charge, is_charged, "simulation.log"
        )
    if not os.path.exists(f"{directory}/MIX/mbar.json"):
        run_mixed_simulations(
            directory, molecule, total_charge, is_charged, "simulation.log"
        )
    if not simulate_all_at_once:
        if not os.path.exists(f"{directory}/water/mbar.json") or not os.path.exists(
            f"{directory}/MIX/mbar.json"
        ):
            wait_for_simulation_results(directory, 2 * steps)
            calculate_free_energy(f"{directory}/water", steps)
            # calculate_free_energy(f"{directory}/MIX", steps)
            wait_for_mbar_results(f"{directory}/water", f"{directory}/MIX")
        water_mbar = get_free_energy(Path(directory) / "water")
        mix_mbar = get_free_energy(Path(directory) / "MIX")
        print(f"Water: {water_mbar:.3f}, MIX: {mix_mbar:.3f}")
        if mix_mbar > water_mbar:
            if not os.path.exists(f"{directory}/DIPC/mbar.json"):
                run_dipc_center_simulations(
                    directory, molecule, total_charge, is_charged, "simulation.log"
                )
            if not os.path.exists(f"{directory}/MIX_interface/mbar.json"):
                run_mixed_interface_simulations(
                    directory, molecule, total_charge, is_charged, "simulation.log"
                )
            if not os.path.exists(f"{directory}/DIPC/mbar.json") or not os.path.exists(
                f"{directory}/MIX_interface/mbar.json"
            ):
                wait_for_simulation_results(f"{directory}", steps * 4)
                calculate_free_energy(f"{directory}/DIPC", steps)
                wait_for_mbar_results(f"{directory}/DIPC", f"{directory}/MIX_interface")
            dipc_mbar = get_free_energy(Path(directory) / "DIPC")
            mix_interface_mbar = get_free_energy(Path(directory) / "MIX_interface")
            print(f"DIPC: {dipc_mbar:.3f}, MIX_interface: {mix_interface_mbar:.3f}")
            if mix_mbar > mix_interface_mbar:
                print(
                    f"Final result: {(water_mbar - dipc_mbar) - (water_mbar - mix_mbar)}"
                )
    else:
        if not os.path.exists(f"{directory}/DIPC/mbar.json"):
            run_dipc_center_simulations(
                directory, molecule, total_charge, is_charged, "simulation.log"
            )
        if not os.path.exists(f"{directory}/MIX_interface/mbar.json"):
            run_mixed_interface_simulations(
                directory, molecule, total_charge, is_charged, "simulation.log"
            )
        if (
            not os.path.exists(f"{directory}/DIPC/mbar.json")
            or not os.path.exists(f"{directory}/MIX_interface/mbar.json")
            or not os.path.exists(f"{directory}/water/mbar.json")
            or not os.path.exists(f"{directory}/MIX/mbar.json")
        ):
            wait_for_simulation_results(f"{directory}", steps * 4)
            calculate_free_energy(f"{directory}/water", steps)
            wait_for_mbar_results(
                f"{directory}/water",
                f"{directory}/DIPC",
                f"{directory}/MIX",
                f"{directory}/MIX_interface",
            )
        water_mbar = get_free_energy(Path(directory) / "water")
        mix_mbar = get_free_energy(Path(directory) / "MIX")
        dipc_mbar = get_free_energy(Path(directory) / "DIPC")
        mix_interface_mbar = get_free_energy(Path(directory) / "MIX_interface")
        print(f"Water: {water_mbar:.3f}, MIX: {mix_mbar:.3f}")
        print(f"DIPC: {dipc_mbar:.3f}, MIX_interface: {mix_interface_mbar:.3f}")
        if mix_mbar > water_mbar and mix_mbar > mix_interface_mbar:
            print(f"Final result: {(water_mbar - dipc_mbar) - (water_mbar - mix_mbar)}")
    print(f'Finished calculations for {db_molecule["name"]}')


def extend_group_list_by_neighbors(
    groups: set[str] | list[str], number_of_neighbors: int = 1
):
    """
    Extend a space separated group list by neighbors. The dimension is determined automatically.
    E.g. if the group is "0 0" and we add one neighbor, the resulting group list will be
    ["-1 -1", "-1 0", "-1 1", "0 -1", "0 0", "0 1", "1 -1", "1 0", "1 1"].
    :param groups: The groups to extend.
    :param number_of_neighbors: The number of neighbors to add.
    :return: The extended group list.
    """
    dimension = len(next(iter(groups)).split())
    range_list = range(-number_of_neighbors, number_of_neighbors + 1)
    neighbor_groups = np.array(list(product(range_list, repeat=dimension)))
    result_groups = set()
    for group in groups:
        group_split = np.array(group.split(), dtype=int)
        group_split = group_split + neighbor_groups
        result_group = [" ".join(map(str, g)) for g in group_split]
        result_groups.update(result_group)
    return list(result_groups)


def get_col(list_of_dicts: list[dict[str, Any]], key: str) -> list[Any]:
    """
    Get a column from a list of dictionaries.
    :param list_of_dicts: The list of dictionaries.
    :param key: The key to get from the dictionaries.
    :return: The list of values.
    """
    return [d[key] for d in list_of_dicts]
