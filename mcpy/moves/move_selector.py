import numpy as np
from mcpy.utils import RandomNumberGenerator


class MoveSelector:
    """Class used to randomly select a Monte Carlo Move from
    a list of Moves.

    Parameters:

    probabilities: A list of probabilities with which the different
        moves should be selected. The norm of this list
        does not need to be 1.

    move_list: The list of moves to select from.

    seed: Seed for reproducibility. Default is None.
    """

    def __init__(self, probabilities, move_list, seed=None):
        assert len(probabilities) == len(move_list)
        self.move_list = move_list
        self.n_moves = sum(probabilities)
        self.rho = np.cumsum(probabilities)
        self.rng = RandomNumberGenerator(seed=seed)
        self.move_counter = [0 for _ in range(len(probabilities))]
        self.move_acceptance = [0 for _ in range(len(probabilities))]

    def __get_index__(self):
        v = self.rng.get_uniform() * self.rho[-1]
        for i in range(len(self.rho)):
            if self.rho[i] > v:
                return i

    def do_trial_move(self, atoms):
        """Choose operator and use it on the candidate. """
        self.to_use = self.__get_index__()
        self.move_counter[self.to_use] += 1
        return self.move_list[self.to_use].do_trial_move(atoms)

    def get_operator(self):
        """Choose operator and return it."""
        to_use = self.__get_index__()
        return self.move_list[to_use]

    def acceptance_counter(self):
        self.move_acceptance[self.to_use] += 1

    def get_acceptance_ration(self):
        ratio = [self.move_acceptance[i]/self.move_counter[i] for i in range(
            len(self.move_acceptance))
                ]
        return ratio

    def reset_counters(self):
        self.move_counter = [0 for _ in range(len(self.move_list))]
        self.move_acceptance = [0 for _ in range(len(self.move_list))]
