from ase import Atoms
import numpy as np

from .base_move import BaseMove
from ..cell import NullCell


class DisplacementMove(BaseMove):
    """Class for performing a displacement move."""

    def __init__(self,
                 species: list[str],
                 seed: int,
                 constraints: list = [],
                 max_displacement: float = 0.1
                 ) -> None:
        """
        Initializes the displacement move with a maximum displacement.

        Parameters:
        max_displacement (float): Maximum displacement distance.
        """
        cell = NullCell()
        super().__init__(cell, species, seed)
        self.max_displacement = max_displacement
        self.constraints = constraints

    def do_trial_move(self, atoms) -> Atoms:
        """
        Displace a random atom by a random vector within the maximum displacement range.

        Returns:
        Atoms: Updated ASE Atoms object after the displacement.
        """
        atoms_new = atoms.copy()
        if len(atoms_new) == 0:
            raise ValueError("No atoms to displace.")
        to_move = np.setdiff1d(np.arange(len(atoms_new)), self.constraints)
        atom_index = self.rng.random.choice(to_move)

        rsq = 1.1
        while rsq > 1.0:
            rx = 2 * self.rng.get_uniform() - 1.0
            ry = 2 * self.rng.get_uniform() - 1.0
            rz = 2 * self.rng.get_uniform() - 1.0
            rsq = rx * rx + ry * ry + rz * rz

        displacement = [rx*self.max_displacement,
                        ry*self.max_displacement,
                        rz*self.max_displacement]

        atoms_new.positions[atom_index] += displacement
        return atoms_new, 0, 'X'
