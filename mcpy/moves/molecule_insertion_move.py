import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from .base_move import BaseMove
from .molecule_utils import find_molecules, random_rotation_matrix
from ..cell import Cell

_MAX_INSERT_ATTEMPTS = 1000


class MoleculeInsertionMove(BaseMove):
    """Insert a rigid molecule at a random position and orientation.

    Mutates ``atoms`` in place by appending the whole molecule; the ensemble
    rolls back on rejection via its arrays snapshot. The template is centered
    on its center of mass at construction and inserted rigidly (uniform
    random rotation, target COM drawn from the cell).

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
        self.molecule.positions -= self.molecule.get_center_of_mass()
        self._template_symbols = sorted(self.molecule.get_chemical_symbols())
        self.min_insert = min_insert
        self.max_molecules = max_molecules
        self._min_insert_sq = min_insert ** 2 if min_insert is not None else None
        self.last_exchange_count = None

    def do_trial_move(self, atoms) -> Atoms:
        """Insert one molecule. Returns the same ``atoms`` (mutated), +1, name."""
        n_mol = len(find_molecules(atoms, self._template_symbols, self.cell))
        if self.max_molecules is not None and n_mol >= self.max_molecules:
            return False, 1, self.name
        self.last_exchange_count = n_mol

        insert_position = self.cell.get_random_point()
        rotation = random_rotation_matrix(self.rng)

        if self._min_insert_sq is not None:
            indices = self.cell.get_atoms_specie_inside_cell(
                atoms, self.cell.get_species()
            )
            if len(indices) > 0:
                positions_bias = atoms.positions[indices]
                min_sq = self._min_insert_sq
                use_mic = bool(np.any(np.asarray(atoms.pbc)))
                placed = False
                for _ in range(_MAX_INSERT_ATTEMPTS):
                    frag_positions = self.molecule.positions @ rotation.T + insert_position
                    diff = (positions_bias[:, None, :] -
                            frag_positions[None, :, :]).reshape(-1, 3)
                    if use_mic:
                        diff = find_mic(diff, atoms.cell, pbc=atoms.pbc)[0]
                    d2 = np.einsum('ij,ij->i', diff, diff)
                    if d2.min() >= min_sq:
                        placed = True
                        break
                    insert_position = self.cell.get_random_point()
                    rotation = random_rotation_matrix(self.rng)
                # Cell too packed to honour min_insert: report a failed move
                # rather than insert a guaranteed overlap (wasted energy eval).
                if not placed:
                    return False, 1, self.name

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
