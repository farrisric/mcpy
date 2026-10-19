from .base_move import BaseMove, fixed_mask, free_atom_count
from ..cell import Cell


class DeletionMove(BaseMove):
    """Class for performing a deletion move.

    Mutates ``atoms`` in place by removing one atom. The ensemble snapshots
    the arrays dict before the move and rolls back on rejection.

    ``min_atoms`` floors the free atoms (``molecule_id < 0``) of the chosen
    species in the whole system, substrate included.
    """
    def __init__(self,
                 cell: Cell,
                 species: list[str],
                 seed: int,
                 min_atoms: int | None = None):
        super().__init__(cell, species, seed)
        self.min_atoms = min_atoms
        self._trial_volume = None

    def do_trial_move(self, atoms) -> int:
        """
        Delete a random atom of the selected species from inside the cell.
        Returns (False, -1, species) if no candidate atom exists or if deleting
        would leave fewer than ``min_atoms`` of that species (no mutation).
        """
        selected_species = self.rng.random.choice(self.species)

        if self.min_atoms is not None:
            if free_atom_count(atoms, selected_species) <= self.min_atoms:
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
        # FixAtoms-held atoms are not candidates either: insertions only
        # append free atoms, so deleting one could never be reversed.
        indices_of_species = indices_of_species[
            ~fixed_mask(atoms)[indices_of_species]]
        if len(indices_of_species) == 0:
            return False, -1, 'X'
        remove_index = int(self.rng.random.choice(indices_of_species))
        del atoms[remove_index]
        self._trial_volume = self.cell.free_volume(atoms)
        return atoms, -1, selected_species

    def get_volume(self) -> float:
        """Free volume of the post-deletion state of the last trial: the
        volume the reverse insertion would be proposed in. Exact detailed
        balance needs this, not the volume of the state still holding the
        deleted particle (docs/gcmc_acceptance_convention.rst)."""
        return self._trial_volume
