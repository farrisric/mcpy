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
                 seed: int,
                 min_atoms: int | None = None):
        super().__init__(cell, species, seed)
        self.min_atoms = min_atoms

    def do_trial_move(self, atoms) -> int:
        """
        Delete a random atom of the selected species from inside the cell.
        Returns (False, -1, species) if no candidate atom exists or if deleting
        would leave fewer than ``min_atoms`` of that species (no mutation).
        """
        selected_species = self.rng.random.choice(self.species)

        if self.min_atoms is not None:
            n_species = atoms.get_chemical_symbols().count(selected_species)
            if n_species <= self.min_atoms:
                return False, -1, selected_species

        indices_of_species = self.cell.get_atoms_specie_inside_cell(
            atoms, selected_species
        )
        # Coexistence with molecule moves: only free atoms (molecule_id < 0)
        # are atomic-deletion candidates; deleting a member atom would corrupt
        # the molecule bookkeeping (its partner would leak, undeletable).
        mol_ids = atoms.arrays.get('molecule_id')
        if mol_ids is not None and len(indices_of_species) > 0:
            indices_of_species = indices_of_species[
                mol_ids[indices_of_species] < 0
            ]
        if len(indices_of_species) == 0:
            return False, -1, 'X'
        remove_index = int(self.rng.random.choice(indices_of_species))
        del atoms[remove_index]
        return atoms, -1, selected_species
