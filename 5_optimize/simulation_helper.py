"""Manually run simulations for molecules."""

import os
import json
from pathlib import Path
from pymongo import MongoClient
from chespex.molecules import Molecule

from optimize_helper import (
    run_water_simulations,
    run_dipc_center_simulations,
    run_mixed_interface_simulations,
    run_mixed_simulations,
    get_free_energy,
)

N_BEADS = 4


def run_molecule_simulations(molecule, level, directory):
    """
    Run all required simulations for a molecule. This includes a water simulation
    and a simulation in a mixed phospholipid bilayer consisting of DIPC and DPPC.
    :param molecule: The name of the molecule from the database.
    :param level: The level of the molecule in the database.
    :param directory: The directory where the simulation files are stored.
    """
    os.makedirs(directory, exist_ok=True)
    # Open database
    client = MongoClient("mongodb://localhost:27017")
    database = client.get_database(f"molecules-{N_BEADS}")
    collection = database.get_collection(f"level-{level}")
    # Load molecule
    db_molecule = collection.find_one({"name": molecule})
    feature_names = {"size": int, "class": int, "charge": int, "oco_w_tfe": float}
    molecule = Molecule.reconstruct(
        db_molecule["bead_names"],
        db_molecule["node_features"],
        db_molecule["edge_index"],
        feature_names,
    )
    total_charge = sum([bead["charge"] - 1 for bead in molecule.beads])
    is_charged = sum([abs(bead["charge"] - 1) for bead in molecule.beads]) > 0
    # Replace bead types
    if level == 0 or level == 1:
        with open("../bead_types/mapping.json", "r", encoding="utf-8") as mapping_file:
            mapping = json.load(mapping_file)[level]
        mapping = {
            bt_name: f'{["K","L"][level]}{mi+1}'
            for mi, bt_name in enumerate(mapping.keys())
        }
        for bead in molecule.beads:
            for btmo, btmn in mapping.items():
                bead["name"] = bead["name"].replace(btmo, btmn)
    ready = True
    if not (Path(directory) / "water" / "mbar.json").exists():
        run_water_simulations(directory, molecule, total_charge, is_charged)
        ready = False
    if not (Path(directory) / "MIX" / "mbar.json").exists():
        run_mixed_simulations(directory, molecule, total_charge, is_charged)
        ready = False
    if ready:
        mix_mbar = get_free_energy(Path(directory) / "MIX")
        water_mbar = get_free_energy(Path(directory) / "water")
        if mix_mbar > water_mbar:
            if not (Path(directory) / "DIPC" / "mbar.json").exists():
                run_dipc_center_simulations(
                    directory, molecule, total_charge, is_charged
                )
            if not (Path(directory) / "MIX_interface" / "mbar.json").exists():
                run_mixed_interface_simulations(
                    directory, molecule, total_charge, is_charged
                )


def main():
    """
    Main function, that runs all simulations for which a directory exists in the
    simulations directory.
    """
    allowed_level_dirs = ["level-0", "level-1", "level-2"]
    molecule_list = []
    parent = Path("simulations")
    for level_name in sorted(parent.iterdir()):
        if str(level_name).count("-") != 1 or level_name.name not in allowed_level_dirs:
            continue
        level = int(str(level_name).rsplit("-", maxsplit=1)[-1])
        for molecule in sorted(level_name.iterdir()):
            in_molecule_list = len(molecule_list) == 0 or molecule.name in molecule_list
            if not molecule.is_dir() or not in_molecule_list:
                continue
            print(f"Running calculations for {molecule}")
            molecule_name = molecule.name.replace("_", " ")
            run_molecule_simulations(molecule_name, level, str(molecule))


if __name__ == "__main__":
    main()
