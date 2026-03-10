<!-- [![GitHub release](https://img.shields.io/github/release/yourusername/npl.svg)](https://GitHub.com/yourusername/npl/releases/) -->

[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/mit)
[![GitHub issues](https://img.shields.io/github/issues/farrisric/npl.svg)](https://GitHub.com/farrisric/mcpy/issues)
[![Documentation Status](https://readthedocs.org/projects/nplib/badge/)](https://mc-py.readthedocs.io/en/latest/index.html)

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

- Grand Canonical Monte Carlo simulations
- Integration with ASE for atomic simulations
- Support for MACE calculator potential and other calculators
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

Please note that `mpi4py` should be installed using conda:

```sh
conda install mpi4py
```

## Dependencies

The package requires the following dependencies:

- `ase>=3.23.0`
- `mace-torch>=0.3.9`
- `mpi4py>=4.0.1`

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

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
