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
                 max_displacement: float = 0.5,
                 max_angle: float = None) -> None:
        """``max_angle`` (radians) caps the rotation per trial: a uniform
        random axis with a uniform angle in [0, max_angle] — still a
        symmetric proposal. ``None`` keeps the full uniform SO(3) rotation,
        which is too aggressive for strongly anchored adsorbates (measured
        ~5% acceptance for CO on CuPd vs ~60% for physisorbed H2O)."""
        super().__init__(cell, [name], seed)
        self.name = name
        self._template_symbols = sorted(molecule.get_chemical_symbols())
        self.max_displacement = max_displacement
        self.max_angle = max_angle

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
        if self.max_angle is None:
            rotation = random_rotation_matrix(self.rng)
        else:
            rotation = self._capped_rotation()

        # Rotate about the COM using minimum-image member offsets so a
        # molecule split across a periodic boundary stays rigid.
        offsets = atoms.positions[members] - com
        if np.any(np.asarray(atoms.pbc)):
            offsets = find_mic(offsets, atoms.cell, pbc=atoms.pbc)[0]
        atoms.positions[members] = com + shift + offsets @ rotation.T
        return atoms, 0, self.name

    def _capped_rotation(self):
        """Rotation by a uniform angle in [0, max_angle] about a uniform
        random axis. Symmetric: the inverse (same angle, flipped axis) is
        proposed with equal probability."""
        while True:
            axis = np.array([self.rng.get_gaussian() for _ in range(3)])
            norm = np.linalg.norm(axis)
            if norm > 1e-12:
                break
        axis /= norm
        angle = self.rng.get_uniform() * self.max_angle
        w = np.cos(angle / 2.0)
        x, y, z = np.sin(angle / 2.0) * axis
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
