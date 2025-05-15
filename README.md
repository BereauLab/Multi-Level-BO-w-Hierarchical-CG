# Multi-level BO with Hierarchical Coarse-Graining

This repository contains the source code for the paper **"Navigating Chemical Space: Multi-Level Bayesian Optimization with Hierarchical Coarse-Graining"**.

### Citing this work
If you find this work useful in your research, please consider citing:

```bibtex
@misc{Walter2025,
  doi = {10.48550/ARXIV.2505.04169},
  url = {https://arxiv.org/abs/2505.04169},
  author = {Walter,  Luis J. and Bereau,  Tristan},
  title = {Navigating Chemical Space: Multi-Level Bayesian Optimization with Hierarchical Coarse-Graining},
  publisher = {arXiv},
  year = {2025},
}
```

## Overview

The following presents the main components of the repository:
- **[chespex](chespex)**: A small Python package that helps with the implementation of the multi-level Bayesian optimization (BO) algorithm for coarse-grained (CG) molecules. It does not contain application-specific details, but rather provides the general framework for the algorithm. It includes the following components:
  - [molecules](chespex/chespex/molecules): This module contains a [`Molecule`](chespex/chespex/molecules/molecule.py) and a [`MoleculeGenerator`](chespex/chespex/molecules/molecule_generator.py) class. The `Molecule` class is used to represent a molecule and its properties, while the `MoleculeGenerator` class is used to enumerate all possible molecular graphs for a given number of CG beads.
  - [encoding](chespex/chespex/encoding): This module contains the model for the [`Encoder`](chespex/chespex/encoding/encoder.py) and [`Decoder`](chespex/chespex/encoding/decoder.py) of the molecular [`Autoencoder`](chespex/chespex/encoding/autoencoder.py). It also contains the permutation invariant [loss function](chespex/chespex/encoding/loss.py) used to train the autoencoder.
  - [optimization](chespex/chespex/optimization): This module includes implementation of a simple [`Dataframe`](chespex/chespex/optimization/data_frame.py) class that is used to store the data for the optimization process. It also includes a [GPyTorch](https://gpytorch.ai/) based [`GaussianProcess`](chespex/chespex/optimization/gaussian_process.py) class used for the Bayesian optimization.
  - [simulation](chespex/chespex/simulation): This module contains simulation utilities for a high-throughput molecular dynamics (MD) simulation workflow. It includes a [`Simulation`](chespex/chespex/simulation/simulation.py) and [`SimulationSetup`](chespex/chespex/simulation/simulation.py) class that provide a Python interface to [GROMACS](https://www.gromacs.org/) simulations. Together with the [simulator functions](chespex/chespex/simulation/simulator.py) they implement a simple queueing system to run multiple GROMACS simulations in parallel and/or on different machines.
- Application specific code in [1_bead_types](1_bead_types), [2_molecule_enumeration](2_molecule_enumeration), [3_autoencoder](3_autoencoder), [4_membrane_setup](4_membrane_setup), and [5_optimize](5_optimize) folders. These folders contain the code used to optimize for a phospholipid phase-separation enhancing small molecule. The numbers indicate the execution sequence. See below for further details.
- [toy-mol-optimization](toy-mol-optimization): This folder contains a Jupyter notebook which was used for the toy example shown in the supplementary information of the paper. It is a simplified (non-invariant) system used to compare the performance of the multi-level BO algorithm with a standard BO algorithm.
- [molecules.json](molecules.json): This JSON file contains the list of molecules which were selected by the multi-level BO algorithm and evaluated via MD simulations. It also lists the results for the demixing free-energy difference, the latent space representation together with the CG resolution level.
- [molecules-single-level.json](molecules-single-level.json): Similar to the above, but for the standard BO algorithm performed at the highest CG resolution only.

## Remarks
- CG molecules are generally treated as graphs with fixed bead-type dependent bond lengths and no angle or dihedral potentials. Molecules are represented as a string by concatenating the bead types and the bond indices. For example, a molecule with bead types `A`, `B`, and `C` and bonds between `A` and `B` and between `B` and `C` is represented as `A B C,0-1 1-2`. The [`Molecule`](chespex/chespex/molecules/molecule.py) class implements a function to convert molecules to this string representation.
- The repository includes neither a list of all enumerated molecules nor their latent space representations due to the size of the files. However, the molecule enumeration can be easily repeated by the [`MoleculeGenerator`](chespex/chespex/molecules/molecule_generator.py) as shown in the [2_molecule_enumeration/1_generate-molecules.ipynb](2_molecule_enumeration/1_generate-molecules.ipynb) notebook. The [3_autoencoder](3_autoencoder) folder contains the trained autoencoder models. [2_molecule_enumeration/2_insert-latent-space.ipynb](2_molecule_enumeration/2_insert-latent-space.ipynb) notebook shows how to generate latent space representations for all enumerated molecules based on the trained autoencoder models.
- The low, medium, and high CG resolution are generally referred to as level 0, level 1, and level 2, respectively.

## Setup

It is recommended to create a new Python environment (e.g. with [conda](https://www.anaconda.com/docs/tools/working-with-conda/environments) or [uv](https://docs.astral.sh/uv/pip/environments/)) with Python version 3.11. The following commands install the helper package `chespex` and other dependencies for the optimization procedure.

```bash
# The following command is only needed if you want to install the CUDA version of PyTorch
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu118
# Install all required packages
pip install -r requirements.txt
```
**Other requirements:**
- [GROMACS](https://www.gromacs.org/) is required to run the MD simulations. We used GROMACS version 2024.2 for all simulations. [This page](https://manual.gromacs.org/documentation/current/install-guide/index.html) describes how to install GROMACS.
- A local [MongoDB](https://www.mongodb.com/docs/manual/installation/) database is used to store and retrieve the enumerated molecules. It is also possible to use a remote MongoDB database. With adapted code, any other database with multiple indexes can be used as well. The indexes are used for a fast retrieval of molecules.


## Optimization procedure

#### Bead type generation &#8594; [1_bead_types](1_bead_types)
- [force-fields.ipynb](1_bead_types/force-fields.ipynb): This notebook generates a modified [`martini.itp`](1_bead_types/martini_mod.itp) force field file with custom bead types for low-resolution CG models. The lower resolution bead types are called K and L for the low and medium resolution models, respectively. This file is required for the GROMACS simulations of lower resolution models.
- [prepare-bead-types.ipynb](1_bead_types/prepare-bead-types.ipynb): This notebook generates the files [`mapping.json`](1_bead_types/mapping.json) and [`bead-types.json`](1_bead_types/bead-types.json) which contain the mapping of the bead types between CG resolutions and bead type features, respectively. These files are used for the enumeration and encoding of the molecules.

#### Molecule enumeration (part 1) &#8594; [2_molecule_enumeration](2_molecule_enumeration)
- [1_generate-molecules.ipynb](2_molecule_enumeration/1_generate-molecules.ipynb): This notebook generates all possible molecular graphs for a given maximum number of CG beads based on the previously generated bead types. The molecules are stored in a MongoDB database.

#### Autoencoder training &#8594; [3_autoencoder](3_autoencoder)
- [training.ipynb](3_autoencoder/training.ipynb): This notebook trains an autoencoder model for the latent space representation of the enumerated molecules. The model is trained on the enumerated molecules in the MongoDB database. Trained models are saved in the [3_autoencoder](3_autoencoder) folder.

#### Molecule enumeration (part 2) &#8594; [2_molecule_enumeration](2_molecule_enumeration)
- [2_insert-latent-space.ipynb](2_molecule_enumeration/2_insert-latent-space.ipynb): This notebook generates the latent space representation of all enumerated molecules using the trained autoencoder models. The latent space representations are stored in the MongoDB database.
- [3_insert-parents.ipynb](2_molecule_enumeration/3_insert-parents.ipynb): This notebook generates the parent-child relationships between the enumerated molecules at different CG resolutions. The parent-child relationships are stored in the MongoDB database for a fast retrieval of the molecules based on their parent or child molecules.
- [4_index-latent-space.ipynb](2_molecule_enumeration/4_index-latent-space.ipynb): This notebook generates a cell list for the latent space and a corresponding index. This cell list is used for a fast retrieval of neighboring molecules in the latent space.

#### Membrane system setup &#8594; [4_membrane_setup](4_membrane_setup)
- [generate_membrane.py](4_membrane_setup/generate_membrane.py): This script uses the program [Insane](https://github.com/Tsjerk/Insane) to generate a membrane system. After the generation, the system is minimized and equilibrated with GROMACS. The script can be called with various arguments:
    ```bash
    usage: generate_membrane.py [-h] [-t {DPPC,DIPC,MIX}] [-s SIZE] [-z HEIGHT] [-d DIRECTORY]
    options:
    -h                    Show help message
    -t {DPPC,DIPC,MIX}    Type of membrane to setup
    -s SIZE               X and Y size of the membrane
    -z HEIGHT             Height of the simulation box
    -d DIRECTORY          Directory to store the membrane files
    ```

#### Molecule optimization &#8594; [5_optimize](5_optimize)
- Generate simulation directories for all single-bead molecules at the lowest CG resolution to obtain a prior for the lowest CG resolution:
    ```bash
    cd 5_optimize
    for m in C N P Q+ Q- SC SN SP SQ+ SQ- TC TN TP TQ+ TQ-; do
        mkdir -p "simulations/level-0/$m,"
    done
    ```
- Run simulations for all single-bead molecules at the lowest CG resolution:
    ```bash
    python simulation_helper.py
    # After the simulations are finished, we calculate the free energies
    python run_mbar.py
    ```
- [initialize.ipynb](5_optimize/initialize.ipynb): This notebook generates the initialization molecules for the optimization.
- Once again, we run the `simulation_helper.py` script to run the simulations for the initialization molecules:
    ```bash
    python simulation_helper.py
    # After the simulations are finished, we calculate the free energies
    python run_mbar.py
    ```
- [optimize.py](5_optimize/optimize.py): This script runs the multi-level Bayesian optimization algorithm. It continues the optimization until interrupted by the user.

#### Single-level optimization with standard BO &#8594; [5_optimize](5_optimize)
- [initialize-single-level.ipynb](5_optimize/initialize-single-level.ipynb): This notebook generates the initialization molecules for the standard BO.
- We run the `simulation_helper.py` script to run the simulations for the initialization molecules:
    ```bash
    python simulation_helper.py
    # After the simulations are finished, we calculate the free energies
    python run_mbar.py level-2
    ```
- [single-level-helper-files/create.py](5_optimize/single-level-helper-files/create.py): Execute this script to generate numpy files of the high resolution latent space. These files are used for a faster acquisition function evaluation in the subsequent optimization step.
- [optimize-single-level.py](5_optimize/optimize-single-level.py): This script runs the standard Bayesian optimization algorithm. It continues the optimization until interrupted by the user.