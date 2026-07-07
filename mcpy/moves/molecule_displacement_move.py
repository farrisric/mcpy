import numpy as np
from ase import Atoms
from ase.geometry import find_mic

from .base_move import BaseMove
from .molecule_utils import find_molecules, molecule_com, random_rotation_matrix
from ..cell import Cell


class MoleculeDisplacementMove(BaseMove):
    """Rigid-body displacement of one molecule.

    Picks a molecule of this species (center of mass inside the cell)
    uniformly, translates its center of mass by a random vector drawn
    uniformly from a sphere of radius ``max_displacement``, and applies a
    uniform random rotation about the (displaced) center of mass. Both parts
    of the proposal are symmetric, so the move enters the standard Metropolis
    branch (``delta_N = 0``); the ensemble rolls back on rejection via its
    arrays snapshot, like every other move.

    This is the mcpy analogue of the MC translation/rotation moves LAMMPS
    ``fix gcmc`` performs between exchanges (its ``M`` argument): it lets an
    adsorbed molecule migrate between sites without a delete/reinsert cycle.
    """

    def __init__(self,
                 cell: Cell,
                 molecule: Atoms,
                 name: str,
                 seed: int,
                 max_displacement: float = 0.5) -> None:
        super().__init__(cell, [name], seed)
        self.name = name
        self._template_symbols = sorted(molecule.get_chemical_symbols())
        self.max_displacement = max_displacement

    def do_trial_move(self, atoms) -> Atoms:
        """Rigidly translate + rotate one molecule. Returns the same
        ``atoms`` (mutated), 0, name; ``(False, 0, name)`` if the cell holds
        no molecule of this species."""
        groups = find_molecules(atoms, self._template_symbols, self.cell)
        if len(groups) == 0:
            return False, 0, self.name
        members = self.rng.random.choice(groups)

        com = molecule_com(atoms, members)
        rsq = 1.1
        while rsq > 1.0:
            rx = 2.0 * self.rng.get_uniform() - 1.0
            ry = 2.0 * self.rng.get_uniform() - 1.0
            rz = 2.0 * self.rng.get_uniform() - 1.0
            rsq = rx * rx + ry * ry + rz * rz
        shift = np.array([rx, ry, rz]) * self.max_displacement
        rotation = random_rotation_matrix(self.rng)

        # Rotate about the COM using minimum-image member offsets so a
        # molecule split across a periodic boundary stays rigid.
        offsets = atoms.positions[members] - com
        if np.any(np.asarray(atoms.pbc)):
            offsets = find_mic(offsets, atoms.cell, pbc=atoms.pbc)[0]
        atoms.positions[members] = com + shift + offsets @ rotation.T
        return atoms, 0, self.name
