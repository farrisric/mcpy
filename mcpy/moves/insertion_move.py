from ase import Atoms
import numpy as np

from .base_move import BaseMove
from ..cell import Cell

_MAX_INSERT_ATTEMPTS = 1000


class InsertionMove(BaseMove):
    """Class for performing an insertion move."""
    def __init__(self,
                 cell : Cell,
                 species: list[str],
                 seed : int,
                 min_insert : float = None) -> None:
        super().__init__(cell, species, seed)
        self.min_insert = min_insert

    def do_trial_move(self, atoms) -> Atoms:
        """
        Insert a random atom of a random species at a random position.

        Returns:
        Atoms: Updated ASE Atoms object after the insertion.
        """
        atoms_new = atoms.copy()
        selected_species = self.rng.random.choice(self.species)
        positions_bias = atoms_new[self.cell.get_atoms_specie_inside_cell(
            atoms_new, self.cell.get_species())].positions

        insert_position = self.cell.get_random_point()

        if self.min_insert is not None and len(positions_bias) > 0:
            for _ in range(_MAX_INSERT_ATTEMPTS):
                dists = np.linalg.norm(positions_bias - insert_position, axis=1)
                if dists.min() >= self.min_insert:
                    break
                insert_position = self.cell.get_random_point()

        atoms_new += Atoms(selected_species, positions=[insert_position])
        return atoms_new, 1, selected_species
