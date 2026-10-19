import numpy as np
from ase import Atoms

from .base_move import BaseMove
from .molecule_utils import find_molecules, random_rotation_matrix
from ..cell import Cell


class MoleculeInsertionMove(BaseMove):
    """Insert a rigid molecule at a random position and orientation.

    Mutates ``atoms`` in place by appending the whole molecule; the ensemble
    rolls back on rejection via its arrays snapshot. The template is centered
    on its center of mass at construction and inserted rigidly (uniform
    random rotation, target COM drawn from the cell).

    The center of mass is uniform over the cell free volume
    (:meth:`mcpy.cell.Cell.get_random_free_point`): the exclusion rule tests
    the COM only, exactly like the free-volume estimate, so the proposal
    density is ``1 / V`` for the ``V`` the acceptance uses. Overlaps of the
    other member atoms are left to the energy. With ``min_insert`` the COM
    only keeps ``min_insert`` from every atom (a relaxation seed, ADR 0004).
    Use ``min_insert`` only with a relaxing calculator: with single-point
    energies it shifts mu by ``kT ln(V_free / V_seed)``.

    Sets ``last_exchange_count`` to the pre-move count of molecules of this
    species with COM inside the cell: the N of the textbook rigid-molecule
    acceptance (see docs/gcmc_acceptance_convention.rst).
    """

    def __init__(self,
                 cell: Cell,
                 molecule: Atoms,
                 name: str,
                 seed: int,
                 min_insert: float = None,
                 max_molecules: int | None = None) -> None:
        super().__init__(cell, [name], seed)
        self.name = name
        self.molecule = molecule.copy()
        # A template cut from an mcpy frame carries molecule_id; the move
        # writes its own.
        self.molecule.arrays.pop('molecule_id', None)
        self.molecule.positions -= self.molecule.get_center_of_mass()
        self._template_symbols = sorted(self.molecule.get_chemical_symbols())
        self.min_insert = min_insert
        self.max_molecules = max_molecules
        self.last_exchange_count = None

    def do_trial_move(self, atoms) -> Atoms:
        """Insert one molecule. Returns the same ``atoms`` (mutated), +1, name."""
        n_mol = len(find_molecules(atoms, self._template_symbols, self.cell))
        if self.max_molecules is not None and n_mol >= self.max_molecules:
            return False, 1, self.name
        self.last_exchange_count = n_mol

        insert_position = self.cell.get_random_free_point(atoms, self.min_insert)
        if insert_position is None:
            return False, 1, self.name
        rotation = random_rotation_matrix(self.rng)

        fragment = self.molecule.copy()
        fragment.positions = fragment.positions @ rotation.T + insert_position

        ids = atoms.arrays.get('molecule_id')
        if ids is None:
            atoms.new_array('molecule_id', np.full(len(atoms), -1, dtype=int))
            next_id = 0
        else:
            next_id = int(ids.max()) + 1 if (ids >= 0).any() else 0
        fragment.new_array('molecule_id', np.full(len(fragment), next_id, dtype=int))
        atoms += fragment
        return atoms, 1, self.name
