from ase import Atoms

from .base_move import BaseMove
from .molecule_utils import find_molecules
from ..cell import Cell


class MoleculeDeletionMove(BaseMove):
    """Delete a whole molecule chosen uniformly from inside the cell.

    Mutates ``atoms`` in place; the ensemble snapshots arrays + constraints
    and rolls back on rejection. Candidates are molecules whose sorted
    member symbols match the template composition and whose mic-aware
    center of mass lies inside the cell.

    Sets ``last_exchange_count`` to the pre-move candidate count (including
    the deleted molecule): the N of the textbook rigid-molecule acceptance
    (see docs/gcmc_acceptance_convention.rst).
    """

    def __init__(self,
                 cell: Cell,
                 molecule: Atoms,
                 name: str,
                 seed: int,
                 min_molecules: int | None = None) -> None:
        super().__init__(cell, [name], seed)
        self.name = name
        self._template_symbols = sorted(molecule.get_chemical_symbols())
        self.min_molecules = min_molecules
        self.last_exchange_count = None

    def do_trial_move(self, atoms) -> Atoms:
        """Delete one molecule. Returns the same ``atoms`` (mutated), -1, name."""
        groups = find_molecules(atoms, self._template_symbols, self.cell)
        if len(groups) == 0:
            return False, -1, self.name
        if self.min_molecules is not None and len(groups) <= self.min_molecules:
            return False, -1, self.name
        self.last_exchange_count = len(groups)
        members = self.rng.random.choice(groups)
        del atoms[members]
        return atoms, -1, self.name
