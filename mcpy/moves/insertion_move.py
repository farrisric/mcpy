import warnings

from ase import Atoms

from .base_move import BaseMove
from ..cell import Cell

_MIN_INSERT_DEPRECATION = (
    'min_insert is deprecated and ignored. Insertions now avoid the exclusion '
    'spheres set by the cell species_radii, the same rule the cell free volume '
    'uses, so the acceptance stays exact. Set a species_radii entry of at least '
    'min_insert for every species instead.')


def _warn_min_insert(min_insert):
    if min_insert is not None:
        warnings.warn(_MIN_INSERT_DEPRECATION, FutureWarning, stacklevel=3)


class InsertionMove(BaseMove):
    """Class for performing an insertion move.

    Mutates ``atoms`` in place by appending one atom. The ensemble is
    responsible for rolling back the mutation on rejection (it snapshots the
    arrays dict before the move).

    The position is uniform over the cell free volume
    (:meth:`mcpy.cell.Cell.get_random_free_point`), so the proposal density
    is ``1 / V`` for the free volume ``V`` the acceptance uses.
    ``min_insert`` is deprecated and ignored: use the cell ``species_radii``.
    """
    def __init__(self,
                 cell: Cell,
                 species: list[str],
                 seed: int,
                 min_insert: float = None,
                 max_atoms: int | None = None) -> None:
        super().__init__(cell, species, seed)
        _warn_min_insert(min_insert)
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

        insert_position = self.cell.get_random_free_point(atoms)
        if insert_position is None:
            return False, 1, selected_species

        atoms += Atoms(selected_species, positions=[insert_position])
        # Coexistence with molecule moves: ASE ``extend`` zero-pads arrays
        # missing from the fragment, which would silently attach this atom to
        # molecule id 0. Tag it as free (-1) explicitly.
        if 'molecule_id' in atoms.arrays:
            atoms.arrays['molecule_id'][-1] = -1
        return atoms, 1, selected_species
