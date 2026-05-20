from ase import Atoms
import numpy as np

from .base_move import BaseMove
from ..cell import Cell

_MAX_INSERT_ATTEMPTS = 1000


class InsertionMove(BaseMove):
    """Class for performing an insertion move.

    Mutates ``atoms`` in place by appending one atom. The ensemble is
    responsible for rolling back the mutation on rejection (it snapshots the
    arrays dict before the move).
    """
    def __init__(self,
                 cell: Cell,
                 species: list[str],
                 seed: int,
                 min_insert: float = None) -> None:
        super().__init__(cell, species, seed)
        self.min_insert = min_insert
        # Squared threshold so the per-attempt loop can skip the sqrt.
        self._min_insert_sq = min_insert ** 2 if min_insert is not None else None

    def do_trial_move(self, atoms) -> Atoms:
        """
        Insert a random atom of a random species at a random position.
        Returns the same ``atoms`` object (mutated) with delta_N=+1.
        """
        selected_species = self.rng.random.choice(self.species)

        insert_position = self.cell.get_random_point()
        if self._min_insert_sq is not None:
            indices = self.cell.get_atoms_specie_inside_cell(
                atoms, self.cell.get_species()
            )
            if len(indices) > 0:
                positions_bias = atoms.positions[indices]
                min_sq = self._min_insert_sq
                for _ in range(_MAX_INSERT_ATTEMPTS):
                    diff = positions_bias - insert_position
                    d2 = np.einsum('ij,ij->i', diff, diff)
                    if d2.min() >= min_sq:
                        break
                    insert_position = self.cell.get_random_point()

        atoms += Atoms(selected_species, positions=[insert_position])
        return atoms, 1, selected_species
