# pylint: disable=invalid-name, unspecified-encoding, missing-module-docstring, missing-function-docstring, nested-min-max
import sys
import signal
from datetime import datetime
import argparse
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
    extend_group_list_by_neighbors,
    get_col,
)


MOLECULES_FILENAME = "molecules.list"


def load_data(filename, completed_molecule_dirs, db):
    data = Dataframe(
        columns=[
            "latent_space",
            "ddg",
            "prior",
            "prediction",
            "molecule_name",
            "level",
            "parent",
            "parent_latent_space",
            "grandparent",
            "grandparent_latent_space",
        ]
    )
    with open(filename, "r") as f:
        molecule_file = f.read().splitlines()
    predictions = {}
    for line in molecule_file:
        line_content = [l.strip() for l in line.split(";")]
        if len(line_content) > 1:
            predictions[line_content[0]] = float(line_content[2])
        else:
            predictions[line_content[0]] = None
    for completed_molecule_dir in completed_molecule_dirs:
        level = int(completed_molecule_dir["level"])
        collection = db.get_collection(f"level-{level}")
        molecule_name = completed_molecule_dir["name"]
        molecule = collection.find_one({"name": molecule_name})
        mbar_result = load_result_for_molecule(completed_molecule_dir["path"], 0)
        latent_space = molecule["latent_space"]
        prior = molecule["prior"] if "prior" in molecule else 0
        parent, parent_latent_space = None, None
        g_parent, g_parent_latent_space = None, None
        if level > 0:
            parent_collection = db.get_collection(f"level-{level-1}")
            parent = molecule["parent"]
            parent_molecule = parent_collection.find_one({"name": parent})
            parent_latent_space = parent_molecule["latent_space"]
            prior = parent_molecule["prior"] if "prior" in parent_molecule else 0
        if level > 1:
            g_parent_collection = db.get_collection(f"level-{level-2}")
            g_parent = parent_molecule["parent"]
            g_parent_molecule = g_parent_collection.find_one({"name": g_parent})
            g_parent_latent_space = g_parent_molecule["latent_space"]
            prior = g_parent_molecule["prior"]
        data.append(
            (
                latent_space,
                mbar_result,
                prior,
                predictions[molecule_name.replace(" ", "_")],
                molecule_name,
                level,
                parent,
                parent_latent_space,
                g_parent,
                g_parent_latent_space,
            )
        )
    return data


def expected_improvement(estimation_mean, estimation_stddev, best_minimum):
    ei_helper = -estimation_mean + best_minimum
    ei_result = ei_helper * norm.cdf(ei_helper / estimation_stddev)
    ei_result += estimation_stddev * norm.pdf(ei_helper / estimation_stddev)
    return ei_result


def upper_confidence_bound(estimation_mean, estimation_stddev, confidence=0.4):
    return -estimation_mean + confidence * estimation_stddev


def get_unknown_indices(data, molecules, level):
    if isinstance(molecules[0], dict):
        molecules = [m["name"] for m in molecules]
    return np.array(
        [
            i
            for i, m in enumerate(molecules)
            if m not in data["molecule_name", "level", level]
        ]
    )


def printd(*args):
    print(datetime.now().strftime("%Y-%m-%d %H:%M"), *args)


def write_to_file(
    molecule_name,
    level,
    ddg_mean,
    ddg_std,
    ei,
    dist,
    lengthscale,
    noise,
    back_switch=False,
):
    with open(MOLECULES_FILENAME, "a") as f:
        f.write(f'{molecule_name.replace(" ", "_"):<36}; {level};')
        f.write(f"{ddg_mean:8.4f};{ddg_std:7.4f};")
        f.write(f"{ei:10.3e};{dist:6.3f};")
        f.write(f"{lengthscale:6.3f};{noise:10.3e}")
        f.write(";S\n" if back_switch else "\n")


class MoleculeOptimization:
    """
    Class for optimizing molecules using multi-level Bayesian optimization.
    """

    def __init__(self, filename):
        self.filename = filename
        client = MongoClient("mongodb://localhost:27017")
        self.database = client.get_database("molecules-4")
        self.collection0 = self.database.get_collection("level-0")
        self.collection1 = self.database.get_collection("level-1")
        self.collection2 = self.database.get_collection("level-2")
        self.collections = [self.collection0, self.collection1, self.collection2]
        self.gp_info = [{}, {}, {}]
        # Load full latent space and prior for lowest level
        self.molecules0 = list(
            self.collection0.find({}, {"latent_space": 1, "prior": 1, "name": 1})
        )
        self.full_latent_space0 = np.array(get_col(self.molecules0, "latent_space"))
        self.full_prior0 = np.array(get_col(self.molecules0, "prior"))
        # Setup signal handler for graceful exit
        self.quit_loop = False
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, _1, _2):
        if not self.quit_loop:
            print(
                "Exiting after this iteration, press Ctrl+C again to exit immediately"
            )
            self.quit_loop = True
        else:
            print("Exiting")
            sys.exit()

    def _calculate_next_resolution_level(self, data: Dataframe) -> int:
        level = data["level"][-1]
        last_levels = data["level"][-3:]
        if data["prediction"][-3] is None:
            return level
        last_errors = np.abs(data["prediction"][-3:] - data["ddg"][-3:])
        if np.all(last_levels == last_levels[0]) and np.all(last_errors < 0.12):
            if last_levels[0] == 0 and len(data.filter("level", 0)) > 80:
                print("Switching to level 1")
                level = 1
            elif last_levels[0] == 1 and len(data.filter("level", 1)) > 50:
                print("Switching to level 2")
                level = 2
        return level

    def load_data(self) -> Dataframe:
        completed_molecule_dirs = load_completed_molecules_list(self.filename)
        wait_for_simulations_finished(completed_molecule_dirs)
        data = load_data(self.filename, completed_molecule_dirs, self.database)
        printd(f"Data size: {len(data)}")
        return data

    @staticmethod
    def _select_best_molecule(mean, std, best_ddg, unknown_indices):
        ei = expected_improvement(mean, std, best_ddg)
        ei_unknown = ei[unknown_indices]
        selected_indices = np.nonzero(np.isclose(ei_unknown, ei_unknown.max()))[0]
        selected_indices = unknown_indices[selected_indices]
        print("Number of selected indices:", len(selected_indices))
        selected_index = np.random.choice(selected_indices)
        return selected_index, ei[selected_index]

    @staticmethod
    def _minimum_distance(
        latent_space1: list | np.ndarray, latent_space2: list | np.ndarray
    ) -> float:
        """
        Calculate the minimum distance between latent space points. The distance is
        calculated element wise between the two arrays. The minimum distance is
        returned.
        :param latent_space: The first set of latent space points.
        :param full_latent_space: The second set of latent space points.
        :return: The minimum pairwise distance between the two sets of latent space points.
        """
        return np.min(
            np.linalg.norm(np.array(latent_space1) - np.array(latent_space2), axis=-1)
        )

    def optimization_loop(self):
        while not self.quit_loop:
            self.get_next_suggestion()

    def evaluate_molecule(
        self,
        molecule: dict,
        level: int,
        mean: float,
        std: float,
        ei: float,
        min_dist: float,
    ):
        with open(MOLECULES_FILENAME, "a") as f:
            f.write(f'{molecule["name"].replace(" ", "_"):<36}; {level};')
            f.write(f"{mean:8.4f};{std:7.4f};")
            f.write(f"{ei:10.3e};{min_dist:6.3f};")
            f.write(f"{self.gp_info[0]['lengthscale']:6.3f};")
            f.write(f"{self.gp_info[0]['noise']:10.3e}\n")
        # Start simulations
        simulate_molecule(self.collection0, molecule, level=0)

    def fit_model(self, data: Dataframe, level: int):
        # Fit GP for level 0
        gp0 = GaussianProcess(
            lengthscale_constraint=Interval(0.01, 10), fixed_noise=1.28e-3
        )
        gp0_np = GaussianProcess(
            lengthscale_constraint=Interval(0.01, 10), fixed_noise=1.28e-3
        )
        target0 = np.array(data["ddg", "level", 0])
        prior0 = np.array(data["prior", "level", 0])
        gp0.fit(data["latent_space", "level", 0], target0 - prior0)
        gp0_np.fit(data["latent_space", "level", 0], target0)
        stddev0 = gp0.predict(data["latent_space", "level", 0]).stddev.mean()
        # gp0.covar_module.lengthscale = gp0_np.lengthscale
        self.gp_info[0] = {
            "lengthscale": gp0_np.lengthscale,
            "noise": gp0.noise,
            "stddev": stddev0,
        }
        printd(
            f"Level 0 Lengthscale: {gp0_np.lengthscale:.3f} ",
            f"Noise: {gp0.noise:.3e} (std. {stddev0:.4f})",
        )
        if level == 0:
            return gp0, None, None
        # Fit GP for level 1
        parent_latent_space = data["parent_latent_space", "level", 1]
        parent_prediction = gp0.predict(parent_latent_space).mean.numpy()
        parent_prediction += np.array(data["prior", "level", 1])
        gp1 = GaussianProcess(Interval(0.05, gp0_np.lengthscale), fixed_noise=1.28e-3)
        if len(data.filter("level", 1)) > 10:
            gp1.fit(
                data["latent_space", "level", 1],
                data["ddg", "level", 1] - parent_prediction,
            )
        elif len(data.filter("level", 1)) > 0:
            gp1.fit(
                data["latent_space", "level", 1],
                data["ddg", "level", 1] - parent_prediction,
                fixed_lengthscale=gp0.lengthscale * 0.5,
            )
        stddev1 = 0
        if len(data.filter("level", 1)) > 0:
            stddev1 = gp1.predict(data["latent_space", "level", 1]).stddev.mean()
        printd(
            f"Level 1 Lengthscale: {gp1.lengthscale:.3f} ",
            f"Noise: {gp1.noise:.3e} (std. {stddev1:.4f})",
        )
        self.gp_info[1] = {
            "lengthscale": gp1.lengthscale,
            "noise": gp1.noise,
            "stddev": stddev1,
        }
        if level == 1:
            return gp0, gp1, None
        # Fit GP for level 2
        g_parent_latent_space = data["grandparent_latent_space", "level", 2]
        g_parent_prediction = gp0.predict(g_parent_latent_space).mean.numpy()
        g_parent_prediction += np.array(data["prior", "level", 2])
        parent_latent_space = data["parent_latent_space", "level", 2]
        parent_prediction = gp1.predict(parent_latent_space).mean.numpy()
        gp2 = GaussianProcess(Interval(0.005, gp1.lengthscale), fixed_noise=1.28e-3)
        if len(data.filter("level", 2)) > 10:
            gp2.fit(
                data["latent_space", "level", 2],
                data["ddg", "level", 2] - parent_prediction - g_parent_prediction,
            )
        elif len(data.filter("level", 2)) > 0:
            gp2.fit(
                data["latent_space", "level", 2],
                data["ddg", "level", 2] - parent_prediction - g_parent_prediction,
                fixed_lengthscale=gp1.lengthscale * 0.5,
            )
        stddev2 = 0
        if len(data.filter("level", 2)) > 0:
            stddev2 = gp2.predict(data["latent_space", "level", 2]).stddev.mean()
        printd(
            f"Level 2 Lengthscale: {gp2.lengthscale:.3f} ",
            f"Noise: {gp2.noise:.3e} (std. {stddev2:.4f})",
        )
        self.gp_info[2] = {
            "lengthscale": gp2.lengthscale,
            "noise": gp2.noise,
            "stddev": stddev2,
        }
        return gp0, gp1, gp2

    def predict_molecules(self, query, level, *gps):
        select = {"latent_space": 1, "name": 1}
        if level == 0:
            select["prior"] = 1
        else:
            select["parent"] = 1
        molecules = list(self.collections[level].find(query, select))
        latent_spaces = np.array(get_col(molecules, "latent_space"))
        prediction = gps[level].predict(latent_spaces)
        prediction_mean = prediction.mean.numpy()
        prediction_std = prediction.stddev.numpy()
        names = get_col(molecules, "name")
        if level == 0:
            prediction_mean += np.array(get_col(molecules, "prior"))
            return names, prediction_mean, prediction_std
        else:
            parent_list = get_col(molecules, "parent")
            parent_query = {"name": {"$in": list(set(parent_list))}}
            parent_names, parent_mean, _ = self.predict_molecules(
                parent_query, level - 1, *gps
            )
            parent_map = {name: i for i, name in enumerate(parent_names)}
            parent_indices = [parent_map[p] for p in parent_list]
            parent_mean = parent_mean[parent_indices]
            prediction_mean += parent_mean
            return names, prediction_mean, prediction_std

    def get_next_suggestion(self):
        data = self.load_data()
        level = self._calculate_next_resolution_level(data)
        gp0, gp1, gp2 = self.fit_model(data, level)
        if level == 0:
            unknown_indices0 = get_unknown_indices(data, self.molecules0, 0)
            ddg_estimation0 = gp0.predict(self.full_latent_space0)
            selected_index0, expected_improvement0 = (
                MoleculeOptimization._select_best_molecule(
                    ddg_estimation0.mean.numpy() + self.full_prior0,
                    ddg_estimation0.stddev.numpy(),
                    min(data["ddg", "level", 0]) * 1.01,
                    unknown_indices0,
                )
            )
            self.evaluate_molecule(
                self.molecules0[selected_index0],
                0,
                ddg_estimation0.mean.numpy()[selected_index0]
                + self.full_prior0[selected_index0],
                ddg_estimation0.stddev.numpy()[selected_index0],
                expected_improvement0,
                MoleculeOptimization._minimum_distance(
                    self.molecules0[selected_index0]["latent_space"],
                    data["latent_space", "level", 0],
                ),
            )
        else:
            top_parent_names1 = data.filter("level", 0).sort("ddg")["molecule_name"][
                :30
            ]
            children_molecules1 = list(
                self.collection1.find(
                    {"parent": {"$in": top_parent_names1}},
                    {"group": 1, "latent_space": 1},
                )
            )
            molecule_groups1 = {c["group"] for c in children_molecules1}
            known_molecule_names1 = data["molecule_name", "level", 1]
            known_molecules1 = self.collection1.find(
                {"name": {"$in": known_molecule_names1}}, {"group": 1}
            )
            molecule_groups1.update({m["group"] for m in known_molecules1})
            molecule_groups1 = extend_group_list_by_neighbors(molecule_groups1, 1)
            names1, prediction_mean1, prediction_std1 = self.predict_molecules(
                {"group": {"$in": molecule_groups1}}, 1, gp0, gp1
            )
            best_ddg1 = (
                min(min(data["ddg", "level", 1]), min(data["ddg", "level", 0]))
                if len(data.filter("level", 1)) > 0
                else min(data["ddg", "level", 0])
            )
            unknown_indices1 = get_unknown_indices(data, names1, 1)
            selected_index1, expected_improvement1 = (
                MoleculeOptimization._select_best_molecule(
                    prediction_mean1, prediction_std1, best_ddg1, unknown_indices1
                )
            )
            ddg_mean1 = prediction_mean1[selected_index1]
            ddg_std1 = prediction_std1[selected_index1]
            molecule1 = self.collection1.find_one({"name": names1[selected_index1]})

            # Calculate distance to closest known molecule
            mols_with_info1 = self.collection1.find(
                {"parent": {"$in": data["molecule_name", "level", 0]}},
                {"latent_space": 1},
            )
            coords1 = [mol["latent_space"] for mol in mols_with_info1]
            coords1.extend(data["latent_space", "level", 1])
            min_dist1 = MoleculeOptimization._minimum_distance(
                molecule1["latent_space"], coords1
            )
            if level == 1:
                # Check if molecule is too far away from known data
                if min_dist1 > gp1.lengthscale * 2:
                    printd(
                        f'Suggested molecule {molecule1["name"]} is too far away',
                        f"({min_dist1:.3f}) and parent ({molecule1['parent']}) is not",
                        "known running simulations for level 0",
                    )
                    # Add molecule to molecule list file
                    write_to_file(
                        molecule1["parent"],
                        0,
                        ddg_mean1,
                        ddg_std1,
                        expected_improvement1,
                        min_dist1,
                        gp1.lengthscale,
                        gp1.noise,
                        True,
                    )
                    # Start simulations
                    molecule1_parent = self.collection0.find_one(
                        {"name": molecule1["parent"]}
                    )
                    simulate_molecule(self.collection0, molecule1_parent, level=0)
                    return
                write_to_file(
                    molecule1["name"],
                    1,
                    ddg_mean1,
                    ddg_std1,
                    expected_improvement1,
                    min_dist1,
                    gp1.lengthscale,
                    gp1.noise,
                )
                # Start simulations
                simulate_molecule(self.collection1, molecule1, level=1)
            else:
                ### Obtain molecules to be checked ###
                top_parent_names2 = data.filter("level", 1).sort("ddg")[
                    "molecule_name"
                ][:30]
                children_molecules2 = list(
                    self.collection2.find(
                        {"parent": {"$in": top_parent_names2}},
                        {"group": 1, "latent_space": 1},
                    )
                )
                molecule_groups2 = {c["group"] for c in children_molecules2}
                known_mol_names2 = data["molecule_name", "level", 2]
                known_molecules2 = self.collection2.find(
                    {"name": {"$in": known_mol_names2}}
                )
                molecule_groups2.update({m["group"] for m in known_molecules2})
                molecule_groups2 = extend_group_list_by_neighbors(molecule_groups2, 1)
                names2, prediction_mean2, prediction_std2 = self.predict_molecules(
                    {"group": {"$in": molecule_groups2}}, 2, gp0, gp1, gp2
                )
                best_ddg2 = (
                    min(min(data["ddg", "level", 2]), best_ddg1)
                    if len(data.filter("level", 2)) > 0
                    else best_ddg1
                )
                unknown_indices2 = get_unknown_indices(data, names2, 2)
                selected_index2, expected_improvement2 = (
                    MoleculeOptimization._select_best_molecule(
                        prediction_mean2, prediction_std2, best_ddg2, unknown_indices2
                    )
                )
                ddg_mean2 = prediction_mean2[selected_index2]
                ddg_std2 = prediction_std2[selected_index2]
                molecule2 = self.collection2.find_one({"name": names2[selected_index2]})
                # Check if molecule is too far away from known data
                mols_with_info2 = list(
                    self.collection2.find(
                        {"parent": {"$in": data["molecule_name", "level", 1]}},
                        {"latent_space": 1},
                    )
                )
                coords2 = [mol["latent_space"] for mol in mols_with_info2]
                coords2.extend(data["latent_space", "level", 2])
                min_dist2 = MoleculeOptimization._minimum_distance(
                    molecule2["latent_space"], coords2
                )
                if min_dist2 > gp2.lengthscale * 2:
                    printd(
                        f'Suggested molecule {molecule2["name"]} is too far away',
                        f"({min_dist2:.3f}), running simulations for level 1",
                    )
                    # Add molecule to molecule list file
                    write_to_file(
                        molecule2["parent"],
                        1,
                        ddg_mean2,
                        ddg_std2,
                        expected_improvement2,
                        min_dist2,
                        gp2.lengthscale,
                        gp2.noise,
                        True,
                    )
                    # Start simulations
                    molecule2_parent = self.collection1.find_one(
                        {"name": molecule2["parent"]}
                    )
                    simulate_molecule(self.collection1, molecule2_parent, level=1)
                    return
                ### Run level 2 simulations ###
                # Add molecule to molecule list file
                write_to_file(
                    molecule2["name"],
                    2,
                    ddg_mean2,
                    ddg_std2,
                    expected_improvement2,
                    min_dist2,
                    gp2.lengthscale,
                    gp2.noise,
                )
                # Start simulations
                simulate_molecule(self.collection2, molecule2, level=2)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Optimize molecules")
    argparser.add_argument(
        "-f", "--filename", type=str, default=MOLECULES_FILENAME, help="Filename"
    )
    arguments = argparser.parse_args()
    optimizer = MoleculeOptimization(arguments.filename)
    optimizer.optimization_loop()
