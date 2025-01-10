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
        self.rho = np.cumsum(probabilities)
        self.rng = RandomNumberGenerator(seed=seed)

    def __get_index__(self):
        v = self.rng.get_uniform() * self.rho[-1]
        for i in range(len(self.rho)):
            if self.rho[i] > v:
                return i

    def get_new_individual(self, atoms):
        """Choose operator and use it on the candidate. """
        to_use = self.__get_index__()
        return self.move_list[to_use].do_trial_move()

    def get_operator(self):
        """Choose operator and return it."""
        to_use = self.__get_index__()
        return self.move_list[to_use]
