from ase import Atoms
import numpy as np
from sklearn.metrics import pairwise_distances

from .moves import BaseMove


class BiasInsertionMove(BaseMove):
    """Class for performing an insertion move."""
    def __init__(self,
                 species: list[str],
                 species_bias: list[str],
                 seed : int,
                 operating_box : list[list] = None,
                 z_shift : float = None,
                 min_insert : float = None,
                 max_insert : float = None):
        super().__init__(species, seed)
        self.box = operating_box
        self.z_shift = z_shift
        self.min_insert = min_insert
        self.max_insert = max_insert
        self.species_bias = species_bias

    def do_trial_move(self, atoms) -> Atoms:
        """
        Insert a random atom of a random species at a random position.

        Returns:
        Atoms: Updated ASE Atoms object after the insertion.
        """
        atoms_new = atoms.copy()
        selected_species = self.rng.random.choice(self.species)

        # if selected_species not in self.species_bias:
        #     new_position = self.get_random_position()
        #     while self.get_min_dist(new_position, atoms_new.positions) >= self.min_insert:
        #         new_position = self.get_random_position()
        #     atoms_new += Atoms(selected_species, positions=[new_position])
        #     return atoms_new, 1, selected_species

        positions_bias = atoms_new.positions[atoms_new.symbols == self.species_bias]
        new_position = self.get_random_position()
        while self.get_min_dist(new_position, positions_bias) >= self.min_insert:
            new_position = self.get_random_position()

        atoms_new += Atoms(selected_species, positions=[new_position])
        return atoms_new, 1, selected_species

    def check_distance_criteria(self, atoms_new):
        min_dist = min(atoms_new.get_distances(-1, range(len(atoms_new)-1), mic=True))
        if min_dist > self.max_insert or min_dist < self.min_insert:
            return False
        return True

    def get_random_position(self):
        p = np.array([
            self.box[i]*self.rng.get_uniform() for i in range(3)
            ]).sum(axis=0)
        if self.z_shift:
            p[2] += self.z_shift
        return p

    def get_min_dist(self, p, positions):
        min_dist = np.min(pairwise_distances(
                p.reshape(1, -1), positions
                ).flatten())
        return min_dist
