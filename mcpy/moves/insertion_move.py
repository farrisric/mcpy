from ase import Atoms
from ase.geometry import find_mic
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
                 min_insert: float = None,
                 max_atoms: int | None = None) -> None:
        super().__init__(cell, species, seed)
        self.min_insert = min_insert
        self.max_atoms = max_atoms
        # Squared threshold so the per-attempt loop can skip the sqrt.
        self._min_insert_sq = min_insert ** 2 if min_insert is not None else None

    def do_trial_move(self, atoms) -> Atoms:
        """
        Insert a random atom of a random species at a random position.
        Returns the same ``atoms`` object (mutated) with delta_N=+1.
        """
        selected_species = self.rng.random.choice(self.species)

        if self.max_atoms is not None:
            n_species = atoms.get_chemical_symbols().count(selected_species)
            if n_species >= self.max_atoms:
                return False, 1, selected_species

        insert_position = self.cell.get_random_point()
        if self._min_insert_sq is not None:
            indices = self.cell.get_atoms_specie_inside_cell(
                atoms, self.cell.get_species()
            )
            if len(indices) > 0:
                positions_bias = atoms.positions[indices]
                min_sq = self._min_insert_sq
                # Periodic systems need minimum-image distances: a point near
                # a box face can overlap an atom's periodic image.
                use_mic = bool(np.any(np.asarray(atoms.pbc)))
                placed = False
                for _ in range(_MAX_INSERT_ATTEMPTS):
                    diff = positions_bias - insert_position
                    if use_mic:
                        diff = find_mic(diff, atoms.cell, pbc=atoms.pbc)[0]
                    d2 = np.einsum('ij,ij->i', diff, diff)
                    if d2.min() >= min_sq:
                        placed = True
                        break
                    insert_position = self.cell.get_random_point()
                # Cell too packed to honour min_insert: report a failed move
                # rather than insert a guaranteed overlap (wasted energy eval).
                if not placed:
                    return False, 1, selected_species

        atoms += Atoms(selected_species, positions=[insert_position])
        return atoms, 1, selected_species
