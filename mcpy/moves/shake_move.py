from .base_move import BaseMove
from ase import Atoms
import numpy as np
from ..cell import NullCell


class ShakeMove(BaseMove):
    def __init__(self, r_max: float, seed: int) -> None:
        """
        Initializes the Shake move with the given maximum displacement distance and RNG.
        """
        cell = NullCell()
        super().__init__(cell, species=['X'], seed=seed)
        self.r_max = r_max
        self.name = 'Shake'
        self.rng = np.random.default_rng(seed)

    def do_trial_move(self, atoms: Atoms) -> Atoms:
        """
        Performs the shake move by randomly displacing each atom within a sphere of radius r_max.
        """
        atoms_new = atoms.copy()

        displacements = self.rng.uniform(-1, 1, size=(len(atoms), 3))
        norms = np.linalg.norm(displacements, axis=1, keepdims=True)
        normalized_displacements = displacements / norms  # Normalize to unit vectors
        radii = self.rng.uniform(0, self.r_max, size=(len(atoms), 1)) ** (1/3)  # Uniform in sphere
        new_positions = atoms_new.get_positions() + normalized_displacements * radii

        atoms_new.set_positions(new_positions)
        return atoms_new, 0, 'X'
