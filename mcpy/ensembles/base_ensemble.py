import random
import logging

from abc import ABC
from typing import Optional, List

from ase import Atoms
from ase.calculators.calculator import Calculator

import numpy as np

from ..cell import Cell

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseEnsemble(ABC):
    def __init__(self,
                 atoms: Atoms,
                 cells: List[Cell],
                 units_type: str,
                 calculator: Calculator,
                 user_tag: Optional[str] = None,
                 random_seed: Optional[int] = None,
                 traj_file: str = 'traj_test.traj',
                 trajectory_write_interval: Optional[int] = None,
                 outfile: str = 'outfile.out',
                 outfile_write_interval: int = 10) -> None:
        """
        Base class for ensembles in Monte Carlo simulations.

        Args:
            atoms (Atoms): The initial configuration of the system.
            calculator (Calculator): The calculator object used for energy calculations.
            user_tag (str, optional): A user-defined tag for the ensemble. Defaults to None.
            random_seed (int, optional): The random seed for the random number generator. Defaults
            to None.
            trajectory_write_interval (int, optional): The interval at which to write trajectory
            files. Defaults to None.
            traj_file (str, optional): The file to write trajectory data. Defaults to
            'traj_test.traj'.
            outfile (str, optional): The file to write output data. Defaults to 'outfile.out'.
            outfile_write_interval (int, optional): The interval at which to write output files.
            Defaults to 10.
        """
        self._accepted_trials = 0
        self._step = 0

        self._atoms = atoms
        self._cells = cells
        self._calculator = calculator
        self._user_tag = user_tag

        self._trajectory_write_interval = trajectory_write_interval
        self._outfile = outfile
        self._outfile_write_interval = outfile_write_interval
        self._traj_file = traj_file
        with open(self._traj_file, 'w') as traj_file:
            pass

        # random number generator
        if random_seed is None:
            self._random_seed = random.randint(0, int(1e16))
        else:
            self._random_seed = random_seed
        random.seed(a=self._random_seed)

    @property
    def atoms(self) -> Atoms:
        """ Current configuration (copy). """
        return self._atoms

    @property
    def cells(self) -> Cell:
        """ Current configuration (copy). """
        return self._cells

    @atoms.setter
    def atoms(self, atoms):
        self._atoms = atoms

    @property
    def step(self) -> int:
        """ Current trial step counter. """
        return self._step

    def calculate_cells_volume(self, atoms) -> None:
        """
        Update the volume of the cells.

        Args:
            cells (List[Cell]): List of Cell objects.
        """
        for cell in self.cells:
            cell.calculate_volume(atoms)

    def write_outfile(self, step: int, energy: float) -> None:
        """
        Write the step and energy to the output file.

        Args:
            step (int): The current step.
            energy (float): The energy value.
        """
        try:
            with open(self._outfile, 'a') as outfile:
                outfile.write(f'STEP: {step} ENERGY: {energy}\n')
        except IOError as e:
            logger.error(f"Error writing to file {self._outfile}: {e}")

    def initialize_outfile(self) -> None:
        """
        Initializes the output file by overwriting any existing content and writing a header.
        """
        try:
            with open(self._outfile, 'w') as outfile:
                outfile.write("STEP ENERGY\n")
        except IOError as e:
            logger.error(f"Failed to initialize output file '{self._outfile}': {e}")
            raise

    def write_coordinates(self, atoms: Atoms, energy: float) -> None:
        """
        Write the trajectory file.

        Args:
            atoms (Atoms): The atomic configuration.
        """
        try:
            write_xyz(atoms, energy, self._traj_file)
        except IOError as e:
            logger.error(f"Error writing to trajectory file {self._traj_file}: {e}")

    def compute_energy(self, atoms: Atoms) -> float:
        """
        Computes the potential energy of the given atoms.

        Args:
            atoms (Atoms): The configuration of atoms.

        Returns:
            float: The potential energy.
        """
        return self._calculator.get_potential_energy(atoms)


def write_xyz(atoms, energy, filename):
    """
    Write an XYZ file from an ASE Atoms object, including the cell dimensions.
    Optimized for faster file writing using numpy.

    Args:
        atoms (ase.Atoms): The ASE Atoms object to write.
        filename (str): The path of the XYZ file to write to.
    """
    cell = atoms.get_cell()
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    comment = f"energy={energy:.6f}"
    num_atoms = len(atoms)
    atom_data = np.column_stack((symbols, positions))
    header = f"{num_atoms}\n{comment} "
    cell_str = " ".join(f"{value:.8f}" for row in cell for value in row)
    cell_data = f'Lattice="{cell_str}"'

    with open(filename, 'a') as xyz_file:
        xyz_file.write(header)
        xyz_file.write(cell_data + "\n")
        np.savetxt(xyz_file, atom_data, fmt="%s %s %s %s")
