from .base_move import BaseMove, fixed_mask
from ase import Atoms
import numpy as np
from ..cell import NullCell


class ShakeMove(BaseMove):
    def __init__(self, r_max: float, seed: int) -> None:
        """
        Initialize the Shake move with the maximum displacement distance.
        Each atom is displaced by a vector uniformly distributed inside a
        ball of radius ``r_max``. Atoms fixed by ``atoms.constraints``
        (``FixAtoms``) are not moved. Without ``FixAtoms`` the mass-weighted
        mean displacement is removed, so a free cluster does not random-walk
        out of a fixed region; the proposal stays symmetric.
        """
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.r_max = float(r_max)
        self.name = 'Shake'
        # numpy generator for vectorized draws; seeded for reproducibility.
        self._np_rng = np.random.default_rng(seed)

    def do_trial_move(self, atoms: Atoms):
        """
        Displace each atom by a random vector uniformly distributed inside a
        ball of radius ``r_max``. Mutates ``atoms`` in place. A failed
        proposal when no atom is free to move.
        """
        fixed = fixed_mask(atoms)
        if fixed.all():
            return False, 0, 'X'
        n = len(atoms)
        # Random direction on S^2.
        displacements = self._np_rng.standard_normal(size=(n, 3))
        norms = np.linalg.norm(displacements, axis=1, keepdims=True)
        directions = displacements / norms
        # Uniform-in-ball radius: r_max * u^(1/3) with u ~ U(0,1).
        u = self._np_rng.random(size=(n, 1))
        radii = self.r_max * np.cbrt(u)
        shifts = directions * radii
        if not fixed.any():
            masses = atoms.get_masses()
            shifts -= masses @ shifts / masses.sum()
        # set_positions applies the constraints: FixAtoms atoms stay put.
        atoms.set_positions(atoms.positions + shifts)
        return atoms, 0, 'X'
