import numpy as np

from .base_move import BaseMove
from ..cell import NullCell


class DisplacementMove(BaseMove):
    """Class for performing a displacement move.

    Mutates one atom's position in place. The ensemble rolls back via the
    pre-trial arrays snapshot if the move is rejected.
    """

    def __init__(self,
                 species: list[str],
                 seed: int,
                 constraints: list = None,
                 max_displacement: float = 0.1
                 ) -> None:
        cell = NullCell()
        super().__init__(cell, species, seed)
        self.max_displacement = max_displacement
        self.constraints = np.asarray(constraints, dtype=int) if constraints else np.empty(0, int)
        self._cached_to_move = None
        self._cached_natoms = -1

    def _movable_indices(self, n):
        """Indices that are not constrained, cached on N."""
        if self._cached_natoms != n:
            if self.constraints.size:
                mask = np.ones(n, dtype=bool)
                in_range = self.constraints[(self.constraints >= 0) & (self.constraints < n)]
                mask[in_range] = False
                self._cached_to_move = np.where(mask)[0]
            else:
                self._cached_to_move = np.arange(n)
            self._cached_natoms = n
        return self._cached_to_move

    def do_trial_move(self, atoms):
        """
        Displace a random atom by a random vector within ``max_displacement``.
        """
        n = len(atoms)
        if n == 0:
            raise ValueError("No atoms to displace.")
        to_move = self._movable_indices(n)
        atom_index = int(self.rng.random.choice(to_move))

        rsq = 1.1
        while rsq > 1.0:
            rx = 2.0 * self.rng.get_uniform() - 1.0
            ry = 2.0 * self.rng.get_uniform() - 1.0
            rz = 2.0 * self.rng.get_uniform() - 1.0
            rsq = rx * rx + ry * ry + rz * rz

        atoms.positions[atom_index, 0] += rx * self.max_displacement
        atoms.positions[atom_index, 1] += ry * self.max_displacement
        atoms.positions[atom_index, 2] += rz * self.max_displacement
        return atoms, 0, 'X'
