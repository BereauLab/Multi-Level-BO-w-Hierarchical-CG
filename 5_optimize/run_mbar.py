# pylint: disable=wrong-import-position,missing-module-docstring
import sys
from pathlib import Path
import jax

jax.config.update("jax_enable_x64", True)
from optimize_helper import calculate_free_energy

parent = Path("simulations")

skip = []


def main():
    """
    Main function, that runs MBAR for all simulations for which a directory exists in the
    simulations directory.
    """

    if len(sys.argv) > 1:
        level_name = sys.argv[1]
    else:
        level_name = "level-0"
    molecules = sorted((parent / level_name).iterdir())
    molecules = [m for m in molecules if m.is_dir()]
    runs = []
    for molecule in molecules:
        if len(skip) > 0 and molecule.name in skip:
            continue
        for system in molecule.iterdir():
            beads = molecule.name.split(",")[0]
            steps = 36 if "-" in beads or "+" in beads else 26
            runs.append((str(system), steps))
    for d, s in runs:
        calculate_free_energy(d, s)


if __name__ == "__main__":
    main()
