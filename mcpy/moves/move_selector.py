import numpy as np
from mcpy.utils import RandomNumberGenerator


class MoveSelector:
    """Class used to randomly select a Monte Carlo Move from a list of Moves.

    Tracks two sets of counters:
      * per-interval counters (``move_counter`` / ``move_acceptance``) cleared
        by :meth:`reset_counters` — used for live "recent" acceptance ratios.
      * cumulative counters (``move_counter_total`` /
        ``move_acceptance_total``) — never reset; used by ``finalize_run``.

    Failed trial moves (returning falsy from ``do_trial_move``) are recorded in
    ``move_failed_counter`` / ``move_failed_counter_total`` and excluded from
    the acceptance-ratio denominator so ratios reflect viable attempts only.

    Parameters
    ----------
    probabilities : list[int|float]
        Weights for sampling each move. Sum need not be 1.
    move_list : list
        The moves to sample from.
    seed : int, optional
        Seed for reproducibility.
    """

    def __init__(self, probabilities, move_list, seed=None):
        assert len(probabilities) == len(move_list)
        self.move_list = move_list
        self.move_list_names = [move.__class__.__name__[:3] for move in move_list]
        self.n_moves = sum(probabilities)
        self.rho = np.asarray(np.cumsum(probabilities), dtype=float)
        self._rho_total = float(self.rho[-1])
        self._n_moves_idx = len(self.rho)
        self.rng = RandomNumberGenerator(seed=seed)
        n = len(probabilities)
        self.move_counter = [0] * n
        self.move_acceptance = [0] * n
        self.move_failed_counter = [0] * n
        self.move_counter_total = [0] * n
        self.move_acceptance_total = [0] * n
        self.move_failed_counter_total = [0] * n
        self.to_use = 0

    def __get_index__(self):
        v = self.rng.get_uniform() * self._rho_total
        i = int(np.searchsorted(self.rho, v, side='right'))
        if i >= self._n_moves_idx:
            i = self._n_moves_idx - 1
        return i

    def do_trial_move(self, atoms):
        """Choose a move and apply it. Tracks attempt and (if the move
        returned falsy) records the failure so ratios use viable attempts."""
        self.to_use = self.__get_index__()
        self.move_counter[self.to_use] += 1
        self.move_counter_total[self.to_use] += 1
        result = self.move_list[self.to_use].do_trial_move(atoms)
        # The move can signal "couldn't propose" by returning a falsy
        # atoms object (e.g. DeletionMove returns False when the cell is
        # empty for that species).
        atoms_new = result[0] if isinstance(result, tuple) else result
        if not atoms_new:
            self.move_failed_counter[self.to_use] += 1
            self.move_failed_counter_total[self.to_use] += 1
        return result

    def get_volume(self):
        return self.move_list[self.to_use].get_volume()

    def calculate_volume(self, atoms):
        return self.move_list[self.to_use].calculate_volume(atoms)

    def calculate_volumes(self, atoms):
        for move in self.move_list:
            move.calculate_volume(atoms)

    def get_operator(self):
        to_use = self.__get_index__()
        return self.move_list[to_use]

    def acceptance_counter(self):
        self.move_acceptance[self.to_use] += 1
        self.move_acceptance_total[self.to_use] += 1

    def _ratios(self, accepted, attempted, failed):
        out = []
        for i in range(len(attempted)):
            viable = attempted[i] - failed[i]
            if viable > 0:
                out.append(accepted[i] / viable)
            else:
                out.append(np.nan)
        return out

    def interval_ratios(self):
        """Acceptance ratios since the last :meth:`reset_counters`."""
        return self._ratios(self.move_acceptance, self.move_counter,
                            self.move_failed_counter)

    def total_ratios(self):
        """Cumulative acceptance ratios over the whole run."""
        return self._ratios(self.move_acceptance_total,
                            self.move_counter_total,
                            self.move_failed_counter_total)

    # Backwards-compatible aliases.
    def get_acceptance_ratio(self):
        return self.interval_ratios()

    def get_acceptance_ration(self):  # legacy typo'd name
        return self.interval_ratios()

    def reset_counters(self):
        n = len(self.move_list)
        self.move_counter = [0] * n
        self.move_acceptance = [0] * n
        self.move_failed_counter = [0] * n

    def get_name(self):
        return self.move_list_names[self.to_use]
