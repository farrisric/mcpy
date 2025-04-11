from ase import Atoms
import numpy as np
from sklearn.metrics import pairwise_distances

from .base_move import BaseMove
from ..cell import Cell


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
        min_dist = np.min(pairwise_distances(
            insert_position.reshape(1, -1), positions_bias).flatten())

        while min_dist < self.min_insert:
            insert_position = self.cell.get_random_point()
            min_dist = np.min(pairwise_distances(
                insert_position.reshape(1, -1), positions_bias).flatten())
        atoms_new += Atoms(selected_species, positions=[insert_position])
        return atoms_new, 1, selected_species
