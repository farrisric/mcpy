import numpy as np

from .base_move import BaseMove
from ..cell import NullCell


class PermutationMove(BaseMove):
    """Class for performing a permutation move.

    Mutates ``atoms`` in place by swapping two atomic numbers. The ensemble
    rolls back the arrays snapshot on rejection.
    """
    def __init__(self,
                 species: list[str],
                 seed: int
                 ) -> None:
        cell = NullCell()
        super().__init__(cell, species, seed)

    def do_trial_move(self, atoms):
        """
        Permute the chemical numbers of two random atoms of different species.
        """
        species_pair = self.rng.random.sample(self.species, 2)
        symbols = np.asarray(atoms.get_chemical_symbols())
        indices_a = np.where(symbols == species_pair[0])[0]
        indices_b = np.where(symbols == species_pair[1])[0]
        if len(indices_a) == 0 or len(indices_b) == 0:
            return False, 0, 'X'
        i = int(self.rng.random.choice(indices_a))
        j = int(self.rng.random.choice(indices_b))
        nums = atoms.arrays['numbers']
        nums[i], nums[j] = nums[j], nums[i]
        return atoms, 0, 'X'
