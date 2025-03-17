# mcpy

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
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from ase.calculators.lj import LennardJones
from ase import Atoms
from mcpy.moves import DeletionMove, InsertionMove, DisplacementMove
from mcpy.moves.move_selector import MoveSelector


class Calculator():
    def __init__(self) -> None:
        self.calculator = LennardJones(sigma=3.4, epsilon=0.010086, rc=10.2, smooth=True)

    def get_potential_energy(self, atoms):
        atoms.calc = self.calculator
        return atoms.get_potential_energy()


lj = Calculator()
atoms = Atoms('Ar', cell=[27.2, 27.2, 27.2], pbc=True)
lj.get_potential_energy(atoms)

box = atoms.get_cell()
species = ['Ar']

move_list = [[25, 25, 50],
             [DeletionMove(species=species, seed=12, operating_box=box),
              InsertionMove(species=species, seed=13, operating_box=box),
              DisplacementMove(species=species, seed=14, max_displacement=1.7)]]

move_selector = MoveSelector(*move_list)

Temp = 87.79

gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            calculator=lj,
            mu={'Ar': -8.4*0.010086},
            units_type='metal',
            species=species,
            temperature=Temp,
            move_selector=move_selector,
            outfile=f'T{Temp}.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=f'T{Temp}.xyz')

gcmc.run(1000000)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
