from .moves import BaseMove
from ase import Atoms
import numpy as np


class ShakeMove(BaseMove):
    def __init__(self, r_max: float, seed: int) -> None:
        """
        Initializes the Shake move with the given maximum displacement distance and RNG.
        """
        super().__init__(seed)
        self.r_max = r_max
        self.name = 'Shake'

    def do_trial_move(self, atoms: Atoms) -> Atoms:
        """
        Performs the shake move by randomly displacing each atom within a sphere of radius r_max.
        """
        new_atoms = atoms.copy()

        displacements = self.rng.uniform(-1, 1, size=(len(atoms), 3))
        norms = np.linalg.norm(displacements, axis=1, keepdims=True)
        normalized_displacements = displacements / norms  # Normalize to unit vectors
        radii = self.rng.uniform(0, self.r_max, size=(len(atoms), 1)) ** (1/3)  # Uniform in sphere
        new_positions = new_atoms.get_positions() + normalized_displacements * radii

        new_atoms.set_positions(new_positions)
        return new_atoms
