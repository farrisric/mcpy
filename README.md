[![Tests](https://github.com/farrisric/mcpy/actions/workflows/tests.yml/badge.svg)](https://github.com/farrisric/mcpy/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/mit)
[![GitHub issues](https://img.shields.io/github/issues/farrisric/mcpy.svg)](https://GitHub.com/farrisric/mcpy/issues)
[![Documentation Status](https://readthedocs.org/projects/mc-py/badge/)](https://mc-py.readthedocs.io/en/latest/index.html)

# <span style="font-size:larger;">mcpy</span>

## Table of contents

- [mcpy](#mcpy)
- [About mcpy](#about-mcpy)
- [Features](#features)
- [Installation](#installation)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## About mcpy

`mcpy` is a Python package designed to run atomistic Monte Carlo simulations. It provides tools for performing Grand Canonical Monte Carlo (GCMC) simulations using the Atomic Simulation Environment (ASE) and other dependencies.

## Features

- Basin Hopping, Canonical & Grand Canonical Monte Carlo simulations
- **Molecular adsorbates**: whole-molecule insertion/deletion and rigid
  translate+rotate displacement moves (O2, H2O, CO, NH3, any ASE-buildable
  template), with per-atom `molecule_id` bookkeeping that survives rollback,
  trajectories, and restarts — cross-validated against LAMMPS `fix gcmc`
  (see `benchmark/README.md`)
- Atomic and molecular species can coexist in one simulation (e.g. dissociative
  O at mu_O = mu_O2/2 alongside molecular O2)
- Integration with ASE for atomic simulations
- Support for MACE calculator potential and other calculators
- **Optional NVIDIA Alchemi backend** (`nvalchemi-toolkit`) for GPU-native MACE
  evaluation and FIRE relaxation — **3x speedup**
- Configurable simulation parameters and logging

## Installation

To install the `mcpy` package, ensure you have Python 3.11 or higher and run the following commands:

```sh
pip install .
```

Alternatively, you can install the package in editable mode:

```sh
pip install -e .
```

To use the MACE calculators (and run the bundled examples), add the `mace` extra:

```sh
pip install -e .[mace]
```

### Optional: NVIDIA Alchemi backend (large systems on CUDA)

For GPU-native MACE evaluation on systems with ≥500 atoms, install the optional
`alchemi` extra:

```sh
pip install -e .[alchemi]
```

This pulls in `nvalchemi-toolkit[mace]`. Requires a CUDA-enabled PyTorch build.
See `NVALCHEMI_NOTES.md` for tuning details (keep the defaults
`compile_model=True` and `dt=1.0`; compile costs a one-time warmup and then
speeds up GCMC even though the atom count varies).

#### GPU memory on long runs

GCMC insertions/deletions change the atom count every accepted move, so each
relaxation allocates differently sized tensors. PyTorch's CUDA caching allocator
reserves a fresh block-set per size and pools it, so reserved GPU memory drifts
far above the live footprint over a long run (live stays ~model-sized; reserved
can climb to the card limit and cause spurious OOM).

Launch long runs with expandable segments, which grows one segment in place
instead of a pool per size:

```sh
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python your_run.py
```

This keeps reserved memory near the true per-step peak; live usage is only ever
a small model-sized footprint.

### MPI for replica exchange

`mpi4py` is not pulled in automatically — install it with conda:

```sh
conda install mpi4py
```

## Dependencies

Core:

- `ase>=3.23.0`

Optional:

- `mace-torch>=0.3.9` — MACE calculators (install via `.[mace]`)
- `nvalchemi-toolkit[mace]>=0.1.0` — GPU-native MACE (install via `.[alchemi]`)
- `mpi4py>=4.0.3` — replica exchange (install via conda)

## Usage

Here is an example of how to use the `mcpy` package to run a GCMC simulation:

```python
from ase.cluster import Octahedron
from mace.calculators import mace_mp

from mcpy.moves import DeletionMove
from mcpy.moves import InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
# For large systems on GPU, use AlchemiFCalculator instead:
# from mcpy.calculators import AlchemiFCalculator
from mcpy.cell import SphericalCell


atoms = Octahedron('Ag', 6, 1)

scell = SphericalCell(atoms, vacuum=3, species_radii={'Ag': 2.947, 'O' : 0},
                      mc_sample_points=100_000)

calculator = mace_mp(device='cuda')

species = ['O']

move_list = [[1, 1],
             [DeletionMove(scell,
                           species=['O'],
                           seed=43215423143),
              InsertionMove(scell,
                            species=['O'],
                            min_insert=0.5,
                            seed=3675437856)]]

move_selector = MoveSelector(*move_list)

mus = {'Ag': -2.99, 'O': -4.91}
delta_mu_O = -0.5
mus['O'] += delta_mu_O
T = 500

gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[scell],
            calculator=calculator,
            mu=mus,
            units_type='metal',
            species=species,
            temperature=T,
            move_selector=move_selector)

gcmc.run(1000000)
```

### Molecular GCMC

Whole molecules are exchanged with the reservoir by passing an ASE template:
the chemical potential is the full molecular chemical potential
(`mu = E(molecule) + delta_mu`; orientations are sampled uniformly, so the
rotational partition function is absorbed into `mu` — see
`docs/gcmc_acceptance_convention.rst`).

```python
from ase.build import molecule

from mcpy.moves import (MoleculeInsertionMove, MoleculeDeletionMove,
                        MoleculeDisplacementMove)

co = molecule('CO')
e_co = calculator.get_potential_energy(molecule('CO', cell=[20.0] * 3))

moves = MoveSelector(
    [2, 2, 1],
    [MoleculeInsertionMove(scell, co, 'CO', seed=1, min_insert=1.3),
     MoleculeDeletionMove(scell, co, 'CO', seed=2),
     MoleculeDisplacementMove(scell, co, 'CO', seed=3,
                              max_displacement=0.6, max_angle=0.6)],
)

gcmc = GrandCanonicalEnsemble(
    atoms=atoms, cells=[scell], calculator=calculator,
    mu={'CO': e_co - 0.65},
    molecules={'CO': co},          # registers the molecular species
    units_type='metal', species=[], temperature=400.0,
    move_selector=moves)
```

Runnable examples: `examples/gcmc_molecule_mace.py` (O2 on Ag(111); any g2 molecule via `--molecule`) and
`examples/re_gcmc_co_cupd_batched.py` (CO on a CuPd nanoparticle with batched
replica exchange and a phase diagram). The notebooks in `notebooks/` walk
through the machinery step by step.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup, how to run the tests, and how to submit changes. Bug reports
and feature requests go to the
[issue tracker](https://github.com/farrisric/mcpy/issues).

Run the tests with:

```sh
pip install -e ".[test]"
pytest tests/
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
