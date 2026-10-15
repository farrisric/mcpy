from ase import Atoms

from .base_move import BaseMove
from ..cell import Cell


class InsertionMove(BaseMove):
    """Class for performing an insertion move.

    Mutates ``atoms`` in place by appending one atom. The ensemble is
    responsible for rolling back the mutation on rejection (it snapshots the
    arrays dict before the move).

    Without ``min_insert`` the position is uniform over the cell free volume
    (:meth:`mcpy.cell.Cell.get_random_free_point`), so the proposal density
    is ``1 / V`` for the free volume ``V`` the acceptance uses. With
    ``min_insert`` the position only keeps ``min_insert`` from every atom:
    a seed that the relaxation moves into the ``species_radii`` free volume
    the acceptance uses (ADR 0004).
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

    def do_trial_move(self, atoms) -> Atoms:
        """
        Insert a random atom of a random species at a random free position.
        Returns the same ``atoms`` object (mutated) with delta_N=+1, or
        ``(False, 1, species)`` when the cell has no free point left.
        """
        selected_species = self.rng.random.choice(self.species)

        if self.max_atoms is not None:
            n_species = atoms.get_chemical_symbols().count(selected_species)
            if n_species >= self.max_atoms:
                return False, 1, selected_species

        insert_position = self.cell.get_random_free_point(atoms, self.min_insert)
        if insert_position is None:
            return False, 1, selected_species

        atoms += Atoms(selected_species, positions=[insert_position])
        # Coexistence with molecule moves: ASE ``extend`` zero-pads arrays
        # missing from the fragment, which would silently attach this atom to
        # molecule id 0. Tag it as free (-1) explicitly.
        if 'molecule_id' in atoms.arrays:
            atoms.arrays['molecule_id'][-1] = -1
        return atoms, 1, selected_species
