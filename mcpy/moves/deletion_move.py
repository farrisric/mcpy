from .base_move import BaseMove
from ..cell import Cell


class DeletionMove(BaseMove):
    """Class for performing a deletion move."""
    def __init__(self,
                 cell : Cell,
                 species: list[str],
                 seed : int,):
        super().__init__(cell, species, seed)

    def do_trial_move(self, atoms) -> int:
        """
        Delete a random atom from the structure.

        Returns:
        Atoms: Updated ASE Atoms object after the deletion.
        """
        atoms_new = atoms.copy()
        selected_species = self.rng.random.choice(self.species)
        indices_of_species = self.cell.get_atoms_specie_inside_cell(
            atoms_new, selected_species)
        if len(indices_of_species) == 0:
            return False, -1, 'X'
        remove_index = self.rng.random.choice(indices_of_species)
        del atoms_new[remove_index]
        return atoms_new, -1, selected_species
