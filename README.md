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

class Calculator():
    def __init__(self) -> None:
        self.calculator = LennardJones()

    def get_potential_energy(self, atoms):
        atoms.calc = self.calculator
        return atoms.get_potential_energy()

lj = Calculator()
atoms = Atoms('Ar', cell=[30, 30, 30])
lj.get_potential_energy(atoms)

T = 500
gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            calculator=lj,
            mu={'Ar': -0.5},
            masses={'Ar': 1},
            species=['Ar'],
            temperature=T,
            moves=[1, 1],
            max_displacement=0.2,
            outfile=f'replica_T{T}.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=f'replica_T{T}.xyz',
            min_max_insert=[1.5, 3.0])

gcmc.run(1000000)
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
