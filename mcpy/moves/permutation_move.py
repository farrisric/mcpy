import numpy as np

from .base_move import BaseMove
from ..cell import NullCell


class PermutationMove(BaseMove):
    """Class for performing a permutation move.

    Mutates ``atoms`` in place by swapping two atomic numbers. The ensemble
    rolls back the arrays snapshot on rejection.

    ``n_swaps > 1`` turns the trial into a single compound perturbation:
    ``n_swaps`` independent pair swaps are applied before the energy is
    evaluated — useful for basin-hopping where a single swap typically falls
    back into the same basin after relaxation.
    """
    def __init__(self,
                 species: list[str],
                 seed: int,
                 n_swaps: int = 1,
                 ) -> None:
        if n_swaps < 1:
            raise ValueError(f"n_swaps must be >= 1, got {n_swaps}")
        cell = NullCell()
        super().__init__(cell, species, seed)
        self.n_swaps = int(n_swaps)

    def do_trial_move(self, atoms):
        """
        Permute the chemical numbers of two random atoms of different species,
        repeated ``self.n_swaps`` times in a single trial.
        """
        nums = atoms.arrays['numbers']
        symbols = np.asarray(atoms.get_chemical_symbols())
        for _ in range(self.n_swaps):
            species_pair = self.rng.random.sample(self.species, 2)
            indices_a = np.where(symbols == species_pair[0])[0]
            indices_b = np.where(symbols == species_pair[1])[0]
            if len(indices_a) == 0 or len(indices_b) == 0:
                return False, 0, 'X'
            i = int(self.rng.random.choice(indices_a))
            j = int(self.rng.random.choice(indices_b))
            nums[i], nums[j] = nums[j], nums[i]
            # Refresh symbols view so subsequent swaps see the updated state.
            symbols[i], symbols[j] = symbols[j], symbols[i]
        return atoms, 0, 'X'
