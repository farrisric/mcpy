from .base_move import BaseMove
from ase import Atoms
import numpy as np
from ..cell import NullCell


class ShakeMove(BaseMove):
    def __init__(self, r_max: float, seed: int) -> None:
        """
        Initialize the Shake move with the maximum displacement distance.
        Each atom is displaced by a vector uniformly distributed inside a
        ball of radius ``r_max``.
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
        ball of radius ``r_max``. Mutates ``atoms`` in place.
        """
        n = len(atoms)
        if n == 0:
            return atoms, 0, 'X'
        # Random direction on S^2.
        displacements = self._np_rng.standard_normal(size=(n, 3))
        norms = np.linalg.norm(displacements, axis=1, keepdims=True)
        directions = displacements / norms
        # Uniform-in-ball radius: r_max * u^(1/3) with u ~ U(0,1).
        u = self._np_rng.random(size=(n, 1))
        radii = self.r_max * np.cbrt(u)
        atoms.positions += directions * radii
        return atoms, 0, 'X'
