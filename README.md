[![status](https://joss.theoj.org/papers/b3fcd2a6358f1c92d719c2295a45dc1e/status.svg)](https://joss.theoj.org/papers/b3fcd2a6358f1c92d719c2295a45dc1e)
[![Tests](https://github.com/farrisric/mcpy/actions/workflows/tests.yml/badge.svg)](https://github.com/farrisric/mcpy/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/mit)
[![GitHub issues](https://img.shields.io/github/issues/farrisric/mcpy.svg)](https://GitHub.com/farrisric/mcpy/issues)
[![Documentation Status](https://readthedocs.org/projects/mc-py/badge/)](https://mc-py.readthedocs.io/en/latest/index.html)
[![PyPI](https://img.shields.io/pypi/v/mcpy-gcmc.svg)](https://pypi.org/project/mcpy-gcmc/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farrisric/mcpy/blob/main/notebooks/gcmc_basics_lj.ipynb)

# mcpy

`mcpy` runs Grand Canonical Monte Carlo (GCMC) simulations of atomistic systems on top of the Atomic Simulation Environment (ASE).
It predicts the composition and structure of surfaces and nanoparticles in contact with a gas or metal reservoir, with any ASE calculator: classical potentials, DFT, or machine-learned interatomic potentials such as MACE.

```sh
pip install mcpy-gcmc
```

Documentation: <https://mc-py.readthedocs.io>
Try it in the browser, with no install: [GCMC basics notebook on Colab](https://colab.research.google.com/github/farrisric/mcpy/blob/main/notebooks/gcmc_basics_lj.ipynb).

## Features

- **Grand-canonical Monte Carlo**: atomic insertion and deletion at fixed chemical potential, with the free volume of the insertion region estimated from per-species exclusion radii.
- **Molecular adsorbates**: whole-molecule insertion, deletion and rigid translate+rotate moves (O2, H2O, CO, NH3, any ASE-buildable template), with per-atom `molecule_id` bookkeeping that survives rollback, trajectories and restarts.
  Atomic and molecular species can coexist in one run.
- **Insertion regions** for every common geometry: the periodic box, a window over a slab, a sphere around a nanoparticle, and a dome over a supported particle.
- **Replica exchange** along a temperature or chemical-potential ladder, over MPI or in one process on one GPU with batched energy evaluation.
- **Relaxed trial energies** (the hybrid GCMC scheme of Senftle et al.): each trial can be relaxed before the acceptance test, which makes GCMC workable on dense metal surfaces.
- **Canonical Monte Carlo and basin hopping**: permutation, displacement, shake and Brownian moves, and a running file of the lowest-energy configurations.
- **Optional GPU backend**: MACE evaluation and relaxation (FIRE by default, `optimizer=` for FIRE2 or LBFGS) through NVIDIA's `nvalchemi-toolkit`.
- **Phase diagrams**: post-processing of a chemical-potential sweep into surface and nanoparticle phase diagrams.

## Validation

mcpy's GCMC sampler reproduces LAMMPS `fix gcmc` for atomic and rigid-molecule Lennard-Jones systems, and the ideal gas against its analytic result.
The tables, the rerunnable scripts and a MACE comparison are in [`benchmark/README.md`](benchmark/README.md).

## Installation

mcpy needs Python 3.11 or later.
The distribution is named `mcpy-gcmc` (the `mcpy` name on PyPI belongs to an unrelated package).
You still write `import mcpy`.

```sh
pip install mcpy-gcmc
pip install 'mcpy-gcmc[mace]'   # with an extra, see the table below
```

To run the examples or change the code, install from the repository instead:

```sh
git clone https://github.com/farrisric/mcpy.git
cd mcpy
pip install -e .
```

The base install (ASE, NumPy, SciPy) runs the CPU examples `examples/00` to `examples/02`.
Optional extras:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `.[mace]` | `mace-torch` | `MACE_F_Calculator`, `examples/03` (CPU or GPU) |
| `.[alchemi]` | `nvalchemi-toolkit[mace]` | `AlchemiCalculator`, `AlchemiFCalculator`, `examples/04` and `examples/05` (CUDA GPU) |
| `.[plot]` | `matplotlib` | the phase-diagram utilities |
| `.[mpi]` | `mpi4py` | `ReplicaExchange` over MPI |
| `.[test]` | `pytest` | the test suite |

A conda-installed `mpi4py` (`conda install mpi4py`) is often the safer choice, because it links the MPI library it was built for.
The [installation guide](https://mc-py.readthedocs.io/en/latest/installation.html) covers the GPU backend and the MPI launch.

## Quickstart (CPU, about three minutes)

```sh
python examples/00_hello_gcmc.py
```

The script fills an empty periodic box with Lennard-Jones particles from a reservoir and prints the average particle count next to the LAMMPS value for the same conditions.
Here is the core of it:

```python
from ase import Atoms
from ase.calculators.lj import LennardJones

from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector

atoms = Atoms(cell=[9.0, 9.0, 9.0], pbc=True)   # an empty periodic box
cell = Cell(atoms, seed=6)                      # insert and delete anywhere in it

move_selector = MoveSelector(
    [1, 1, 1],
    [InsertionMove(cell, species=['H'], seed=1),
     DeletionMove(cell, species=['H'], seed=2),
     DisplacementMove(species=['H'], max_displacement=0.3, seed=3)],
    seed=4,
)

gcmc = GrandCanonicalEnsemble(
    atoms=atoms,
    cells=[cell],
    calculator=LennardJones(sigma=1.0, epsilon=1.0, rc=3.0, smooth=False),
    mu={'H': -3.0},          # reduced chemical potential
    species=['H'],
    units_type='LJ',
    temperature=2.0,         # reduced temperature
    move_selector=move_selector,
    random_seed=5,           # every random stream is seeded: the run repeats exactly
    outfile='hello_gcmc.out',
    traj_file='hello_gcmc.xyz',
    outfile_write_interval=100,
    trajectory_write_interval=100,
)
gcmc.run(6000)
```

The [tutorial](https://mc-py.readthedocs.io/en/latest/getting_started.html) takes the same loop to an O/Ag(111) oxidation phase diagram with MACE.
[`examples/`](examples/README.md) has the other runnable scripts, and [`notebooks/`](notebooks/) has worked case studies with executed output.

## Citing mcpy

If you use mcpy in published work, please cite it with the metadata in [`CITATION.cff`](CITATION.cff) (GitHub shows a "Cite this repository" button for it).

## Contributing

Contributions are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the tests, and how to submit changes.
Bug reports and feature requests go to the [issue tracker](https://github.com/farrisric/mcpy/issues).

## License

mcpy is licensed under the MIT License; see [LICENSE](LICENSE).
