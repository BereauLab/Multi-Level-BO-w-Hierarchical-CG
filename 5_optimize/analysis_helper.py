"""Helper functions for visualizations for the paper"""

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from pymongo import MongoClient
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from optimize_helper import load_result_for_molecule

client = MongoClient("mongodb://localhost:27017")
database = client.get_database("molecules-4")

MOLECULE_NUMBERS = {2: 136870880, 1: 6742680, 0: 89960}


def read_xvg(xvg_filename: str) -> np.ndarray:
    """
    Read an XVG file and return the data as a numpy array.
    :param xvg_filename: The filename of the XVG file.
    :return: The data as a numpy array.
    """
    with open(xvg_filename, "r", encoding="utf-8") as file:
        lines = [line for line in file.readlines() if line[0] != "#" and line[0] != "@"]
        data = np.array([list(map(float, line.split())) for line in lines])
        return data


def load_row_result(row: pd.Series):
    """
    Load result for pd row from mbar json files
    :param row: The row of the dataframe.
    """
    level = row["level"]
    name = row["name"]
    if not os.path.exists(f"simulations/level-{level}/{name}"):
        raise FileNotFoundError(
            f"File not found: {f'simulations/level-{level}/{name}'}"
        )
    path = Path(f"simulations/level-{level}/{name}")
    return load_result_for_molecule(path)


def load_molecules_list(filename: str, default_level: int = 0) -> pd.DataFrame:
    """
    Load molecules and their results"
    :param filename: The filename of the molecules list.
    :param default_level: The default level of the molecules
    :return: The molecules as a pandas dataframe.
    """
    columns = [
        "name",
        "level",
        "prediction",
        "stddev",
        "EI",
        "closest distance",
        "lengthscale",
        "noise",
        "switch",
    ]
    converters = {"name": lambda x: x.strip()}
    mols = pd.read_csv(filename, sep=";", names=columns, converters=converters)
    mols["level"] = mols["level"].fillna(default_level).astype(int)
    mols.drop(columns=["switch"], inplace=True)
    mols["result"] = mols.apply(load_row_result, axis=1)
    mols["error"] = mols.apply(lambda row: row["prediction"] - row["result"], axis=1)
    return mols


def load_latent_space_samples(
    level: int, max_number: int = 100_000, attributes: list[str] = None
) -> np.ndarray:
    """
    Load samples from MongoDB
    :param level: The level of the molecules.
    :param max_number: The maximum number of samples to load.
    :param attributes: The attributes to select from the database. If None, only
        the `latent_space` attribute is selected.
    :return: The samples as a numpy
    """
    if attributes is None:
        attributes = ["latent_space"]
    collection = database.get_collection(f"level-{level}")
    molecule_count = MOLECULE_NUMBERS[level]
    attributes = {attr: 1 for attr in attributes}
    sample = list(
        collection.aggregate(
            [{"$sample": {"size": max_number}}, {"$project": attributes}]
        )
        if molecule_count > max_number
        else collection.find({}, attributes)
    )
    if len(attributes) == 1:
        return np.array([sample[list(attributes.keys())[0]] for sample in sample])
    return sample


def get_child_latent_spaces(
    parents: list[str] | pd.Series, parent_level: int
) -> np.ndarray:
    """
    Load child latent spaces from MongoDB
    :param parents: The names of the parents.
    :param parent_level: The level of the parents.
    :return: The child latent spaces as a numpy array.
    """
    if isinstance(parents, pd.Series):
        parents = parents.tolist()
    parents = [name.replace("_", " ") for name in parents]
    collection = database.get_collection(f"level-{parent_level + 1}")
    molecules = collection.find({"parent": {"$in": parents}}, {"latent_space": 1})
    return np.array([molecule["latent_space"] for molecule in molecules])


def get_grandchild_latent_spaces(
    parents: list[str] | pd.Series, parent_level: int
) -> np.ndarray:
    """
    Load grandchild latent spaces from MongoDB
    :param parents: The names of the parents.
    :param parent_level: The level of the parents.
    :return: The grandchild latent spaces as a numpy array.
    """
    if isinstance(parents, pd.Series):
        parents = parents.tolist()
    parents = [name.replace("_", " ") for name in parents]
    collection1 = database.get_collection(f"level-{parent_level + 1}")
    children = collection1.find({"parent": {"$in": parents}}, {"name": 1})
    children_names = [child["name"] for child in children]
    collection2 = database.get_collection(f"level-{parent_level + 2}")
    grandchildren = collection2.find(
        {"parent": {"$in": children_names}}, {"latent_space": 1}
    )
    return np.array([molecule["latent_space"] for molecule in grandchildren])


def get_molecule_priors(molecule: str):
    """
    Load the prior for a molecule from the database
    :param molecule: The name of the molecule.
    :return: The prior.
    """
    collection = database.get_collection("level-0")
    molecule_name = molecule.replace("_", " ")
    molecule = collection.find_one({"name": molecule_name}, {"prior": 1})
    return molecule["prior"]


def load_molecule_latent_space(molecule: pd.Series) -> list[float]:
    """
    Load latent space from MongoDB
    :param molecule: The molecule.
    :return: The latent space as a list.
    """
    collection = database.get_collection(f"level-{molecule['level']}")
    name = molecule["name"].replace("_", " ")
    return collection.find_one({"name": name}, {"latent_space": 1})["latent_space"]


def transform_latent_space(
    latent_space: np.ndarray,
    kde_factor: float,
    padding: float = 0.07,
    steps: int = 300,
    pca: PCA = None,
) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray]:
    """
    Transform latent space using PCA and a gaussian kernel density estimator
    :param latent_space: The latent space.
    :param kde_factor: The factor for the kernel density estimator.
    :param padding: The padding for the latent space.
    :param steps: The number of steps.
    :param pca: The fitted PCA object. If None, a new PCA will be fitted.
    :return: The PCA, x, y, and z.
    """
    if pca is None:
        pca = PCA(n_components=2)
        pca.fit(latent_space)
    transformed_latent_space = pca.transform(latent_space)
    mins, maxs = transformed_latent_space.min(axis=0), transformed_latent_space.max(
        axis=0
    )
    mins, maxs = mins - padding * (maxs - mins), maxs + padding * (maxs - mins)
    x_range = np.linspace(mins[0], maxs[0], steps)
    y_range = np.linspace(mins[1], maxs[1], steps)
    x, y = np.meshgrid(x_range, y_range)
    grid_points = np.vstack([x.ravel(), y.ravel()]).T
    kde = KernelDensity(bandwidth=kde_factor, kernel="gaussian", rtol=1e-5)
    kde.fit(transformed_latent_space)
    log_density = kde.score_samples(grid_points)
    z = np.exp(log_density).reshape(x.shape)
    return pca, x, y, z


def smooth_and_scale(x: np.ndarray, y: np.ndarray, window: int = 500, scale: float = 1):
    """
    Smooth and scale the data
    :param x: The x values.
    :param y: The y values.
    :param window: The window size for the smoothing.
    :param scale: The scale factor.
    :return: The smoothed and scaled x and y values.
    """
    y = np.convolve(y, np.ones(window) / window, "valid")
    x = x[: len(y)] * scale
    return x, y


def parse_colvars_traj(dir_name: Path):
    """ "
    Parse colvars.traj files
    :param dir_name: The directory name.
    :return: The data as a numpy array
    """
    files = [f for f in os.listdir(dir_name) if re.match(r"prod.*\.colvars.traj", f)]
    data = []
    for f in sorted(files):
        with open(f"{dir_name}/{f}", encoding="utf-8") as f:
            lines = [l for l in f.read().splitlines() if not l.startswith("#")]
        data.append(np.array([list(map(float, l.split())) for l in lines]))
    result = data[0]
    for a in data[1:]:
        a[:, 0] += result[-1, 0]
        result = np.vstack((result, a))
    return result


class HandlerTupleVertical(HandlerTuple):
    """
    Custom legend handler that arranges tuple entries below each other.
    """

    def __init__(self, space=5, **kwargs):
        # Custom legend handler that arranges tuple entries below each other.
        self.space = space
        super().__init__(**kwargs)

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Customizes how the tuple handles are displayed in the legend.
        artists = []
        for i, handle in enumerate(orig_handle[::-1]):
            y_offset = ydescent + i * height / 2 + height * 0.2
            new_line = Line2D(
                [width * 0.1, width * 0.9],
                [y_offset, y_offset],
                color=handle.get_color(),
                linestyle=handle.get_linestyle(),
                linewidth=handle.get_linewidth(),
                alpha=handle.get_alpha(),
            )
            new_line.set_transform(trans)
            artists.append(new_line)
        return artists
