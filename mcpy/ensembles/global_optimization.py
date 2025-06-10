from typing import Optional, List, Dict

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from .canonical_ensemble import BaseEnsemble
from .grand_canonical_ensemble import GrandCanonicalEnsemble
from ..utils.random_number_generator import RandomNumberGenerator
from ..utils.set_unit_constant import SetUnits
from ..moves.move_selector import MoveSelector
from ..cell import Cell


class GlobalOptimization(GrandCanonicalEnsemble):
    def __init__(self,
                 atoms: Atoms,
                 cells: List[Cell],
                 units_type: str,
                 calculator: Calculator,
                 mu: Dict[str, float],
                 species: List[str],
                 temperature: float,
                 move_selector: MoveSelector,
                 random_seed: Optional[int] = None,
                 traj_file: str = 'trajectory.xyz',
                 outfile: str = 'outfile.out',
                 outfile_write_interval: Optional[int] = 1) -> None:

        super().__init__(atoms=atoms,
                         cells=cells,
                         units_type='metal',
                         calculator=calculator,
                         random_seed=random_seed,
                         mu=mu,
                         species=species,
                         move_selector=move_selector,
                         temperature=temperature,
                         traj_file=traj_file,
                         outfile=outfile,
                         outfile_write_interval=outfile_write_interval)
        
        self.E_old = self.compute_energy(self.atoms)

        self.units = SetUnits(units_type,
                              temperature=temperature,
                              species=species)
        self.best_en = self.E_old
        self.initial_atoms = len(self.atoms)
        self.n_atoms = len(self.atoms)
        self._temperature = temperature
        self._mu = mu
        self.move_selector = move_selector
        self.calculate_cells_volume(self.atoms)
        self._step = 1
        self.exchange_attempts = 0
        self.exchange_successes = 0
        self.rng_acceptance = RandomNumberGenerator(seed=self._random_seed + 2)

        self.initialize_outfile()

    def write_coordinates(self, atoms: Atoms, energy: float) -> None:
        """
        Write the trajectory file.

        Args:
            atoms (Atoms): The atomic configuration.
        """
        try:
            self.write_xyz(atoms, energy, self._traj_file)
        except IOError as e:
            logger.error(f"Error writing to trajectory file {self._traj_file}: {e}")
    
    def write_xyz(self, atoms, energy, filename):
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
    
        with open(filename, 'w') as xyz_file:
            xyz_file.write(header)
            xyz_file.write(cell_data + "\n")
            np.savetxt(xyz_file, atom_data, fmt="%s %s %s %s")    
    
    def _run(self) -> None:
       """
       Performs a single Monte Carlo step and handles logging and writing.

       Args:
           step (int): The current step.
       """
       self.do_gcmc_step()
       if self._step % self._outfile_write_interval == 0:
           acceptance_ratios = self.move_selector.get_acceptance_ration()
           self.logger.info("{:<10} {:<10} {:<15.6f} {:<20}".format(
               self._step,
               self.n_atoms,
               self.E_old,
               ", ".join(f"{ratio * 100:.1f}%" if not np.isnan(ratio)
                         else "N/A" for ratio in acceptance_ratios)
           ))
           self.write_outfile()
       if self.E_old < self.best_en:
              self.best_en = self.E_old
              self.write_coordinates(self.atoms, self.E_old)

       self._step += 1
