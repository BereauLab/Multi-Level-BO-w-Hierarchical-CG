import sys
import signal
from pathlib import Path
from datetime import datetime
import logging
import numpy as np
from pymongo import MongoClient
from scipy.stats import norm
from chespex.optimization.gaussian_process import GaussianProcess
from chespex.optimization.data_frame import Dataframe
from gpytorch.constraints import Interval
from optimize_helper import (
    simulate_molecule,
    load_result_for_molecule,
    load_completed_molecules_list,
    wait_for_simulations_finished,
)

logging.basicConfig(level=logging.CRITICAL)

MOLECULES_FILENAME = "molecules-single-level.list"
LENGTHSCALE_SCALING = 1.0


def load_data(completed_molecule_dirs, db):
    data = Dataframe(
        columns=[
            "latent_space",
            "ddg",
            "molecule_name",
        ]
    )
    for completed_molecule_dir in completed_molecule_dirs:
        molecule_name = completed_molecule_dir["name"]
        molecule = db.get_collection("level-2").find_one({"name": molecule_name})
        mbar_result = load_result_for_molecule(completed_molecule_dir["path"], 0)
        latent_space = molecule["latent_space"]
        data.append(
            (
                latent_space,
                mbar_result,
                molecule_name,
            )
        )
    return data


def expected_improvement(estimation_mean, estimation_stddev, best_minimum):
    ei_helper = -estimation_mean + best_minimum * 1.01
    ei_result = ei_helper * norm.cdf(ei_helper / estimation_stddev)
    ei_result += estimation_stddev * norm.pdf(ei_helper / estimation_stddev)
    return ei_result


def get_unknown_indices(data, molecule_names):
    known_names = set(data["molecule_name"])
    unkown_indices = [i for i, m in enumerate(molecule_names) if m not in known_names]
    return np.array(unkown_indices)


def select_best_molecule(mean, std, best_ddg, unknown_indices):
    ei = expected_improvement(mean, std, best_ddg)
    ei_unknown = ei[unknown_indices]
    selected_indices = np.nonzero(np.isclose(ei_unknown, ei_unknown.max()))[0]
    selected_indices = unknown_indices[selected_indices]
    selected_index = np.random.choice(selected_indices)
    return selected_index, ei[selected_index]


def min_dist(latent_space, full_latent_space):
    return np.min(np.linalg.norm(full_latent_space - latent_space, axis=-1))


def write_to_file(molecule, level, ddg_mean, ddg_std, ei, dist, lengthscale, noise):
    with open(MOLECULES_FILENAME, "a") as f:
        f.write(f'{molecule["name"].replace(" ", "_"):<36}; {level};')
        f.write(f"{ddg_mean:8.4f};{ddg_std:7.4f};")
        f.write(f"{ei:10.3e};{dist:6.3f};")
        f.write(f"{lengthscale:6.3f};{noise:10.3e}\n")


def printd(*args):
    print(datetime.now().strftime("%Y-%m-%d %H:%M"), *args)


client = MongoClient("mongodb://localhost:27017")
database = client.get_database("molecules-4")
collection = database.get_collection("level-2")


def run_next_iteration():
    completed_molecule_dirs = load_completed_molecules_list(
        MOLECULES_FILENAME, default_level=2
    )
    wait_for_simulations_finished(completed_molecule_dirs)
    data = load_data(completed_molecule_dirs, database)
    printd(f"Data size: {len(data)}")
    ### Fit gp ###
    gp = GaussianProcess(
        lengthscale_constraint=Interval(0.01, 10),
        fixed_noise=1.28e-3,  # Corresponds to a std. dev. of approx. 0.05
    )
    gp.fit(data["latent_space"], data["ddg"])
    lengthscale = gp.lengthscale
    noise = gp.noise
    stddev = gp.predict(data["latent_space"]).stddev.mean()
    printd(
        f"Level 0 Lengthscale: {lengthscale:.3f} Noise: {noise:.3e} (std. {stddev:.4f})"
    )
    best_ddg = min(data["ddg"])
    printd(f"Best ddg: {best_ddg:.4f}")
    best_ei_list = []
    best_results = []
    printd("Start checks")
    latent_space_files = [
        d for d in Path("single-level-helper-files").iterdir() if d.suffix == ".npz"
    ]
    for i, filepath in enumerate(latent_space_files):
        print(i, end="\r")
        latent_space_info = np.load(filepath)
        test_latent_space = latent_space_info["latent_space"]
        test_names = latent_space_info["names"]
        ddg_estimation = gp.predict(test_latent_space)
        prediction_mean = ddg_estimation.mean.numpy()
        prediction_std = ddg_estimation.stddev.numpy()
        unknown_indices = get_unknown_indices(data, test_names)
        selected_index, best_ei = select_best_molecule(
            prediction_mean, prediction_std, best_ddg, unknown_indices
        )
        best_ei_list.append(best_ei)
        best_results.append(
            {
                "name": test_names[selected_index],
                "mean": prediction_mean[selected_index],
                "std": prediction_std[selected_index],
                "latent_space": test_latent_space[selected_index],
            }
        )
    best_molecule = best_results[np.argmax(best_ei_list)]
    molecule_name = best_molecule["name"]
    dist = min_dist(data["latent_space"], best_molecule["latent_space"])
    with open(MOLECULES_FILENAME, "a") as f:
        f.write(f'{molecule_name.replace(" ", "_"):<36}; 2;')
        f.write(f"{best_molecule['mean']:8.4f};{best_molecule['std']:7.4f};")
        f.write(f"{np.min(best_ei_list):10.3e};{dist:6.3f};")
        f.write(f"{lengthscale:6.3f};{noise:10.3e}\n")
    simulate_molecule(
        collection,
        {"name": molecule_name},
        level=2,
        simulate_all_at_once=False,
    )


quit_loop = False


def signal_handler(_1, _2):
    global quit_loop  # pylint: disable=global-statement
    if not quit_loop:
        print("Exiting after this iteration, press Ctrl+C again to exit immediately")
        quit_loop = True
    else:
        print("Exiting")
        sys.exit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    while not quit_loop:
        run_next_iteration()
