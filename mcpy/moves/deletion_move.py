from .base_move import BaseMove
from ..cell import Cell


class DeletionMove(BaseMove):
    """Class for performing a deletion move.

    Mutates ``atoms`` in place by removing one atom. The ensemble snapshots
    the arrays dict before the move and rolls back on rejection.
    """
    def __init__(self,
                 cell: Cell,
                 species: list[str],
                 seed: int):
        super().__init__(cell, species, seed)

    def do_trial_move(self, atoms) -> int:
        """
        Delete a random atom of the selected species from inside the cell.
        Returns (False, -1, 'X') if no candidate atom exists (no mutation).
        """
        selected_species = self.rng.random.choice(self.species)
        indices_of_species = self.cell.get_atoms_specie_inside_cell(
            atoms, selected_species
        )
        if len(indices_of_species) == 0:
            return False, -1, 'X'
        remove_index = int(self.rng.random.choice(indices_of_species))
        del atoms[remove_index]
        return atoms, -1, selected_species
