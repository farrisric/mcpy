from .moves import BaseMove
from ase import Atoms
from ..utils import RandomNumberGenerator
import numpy as np
from sklearn.metrics import pairwise_distances


class InsertionMove(BaseMove):
    """Class for performing an insertion move."""
    def __init__(self,
                 species: list[str],
                 species_bias: list[str],
                 seed : int,
                 operating_box : list[list] = None,
                 z_shift : float = None,
                 min_max_insert : list[float] = None):
        super().__init__(species, seed)
        self.box = operating_box
        self.z_shift = z_shift
        #!self.min_max_insert = min_max_insert
        self.min_insert = min_max_insert[0]
        self.species_bias = species_bias[0]

    def do_trial_move(self, atoms) -> Atoms:
        """
        Insert a random atom of a random species at a random position.

        Returns:
        Atoms: Updated ASE Atoms object after the insertion.
        """
        ### copy old configuration to new configuration
        atoms_new = atoms.copy()
        ### select specie to insert
        selected_species = self.rng.random.choice(self.species)
        ### select position of the atoms from which you add the configurational bias
        elements_array = np.array(atoms_new.get_chemical_symbols())
        positions_bias = atoms_new.positions[elements_array == self.species_bias]
        ### iterate selection of the insertion position until the configurational bias is satisfied
        configurational_bias = False
        while configurational_bias == False:
            insert_position = np.array([
                self.box[i]*self.rng.get_uniform() for i in range(3)
                ]).sum(axis=0)
            ### if z_shift, shift the random position to be in the wanted region
            if self.z_shift:
                insert_position[2] += self.z_shift
            min_dist = np.min(pairwise_distances(insert_position.reshape(1,-1),positions_bias).flatten())    
            if min_dist >= self.min_insert and min_dist <= 3:
                configurational_bias = True         
        ### select specie to insert
        selected_species = self.rng.random.choice(self.species)            
        ### add the new atom in the selected position in the new configuration    
        atoms_new += Atoms(selected_species, positions=[insert_position])
        #!if self.min_max_insert:
        #!    if self.check_distance_criteria(atoms_new):
        #!        return atoms_new, 1, selected_species
        #!    else:
        #!        return False, False, False
        return atoms_new, 1, selected_species

    #!def check_distance_criteria(self, atoms_new):
    #!    min_d = min(atoms_new.get_distances(-1, range(len(atoms_new)-1), mic=True))
    #!    if min_d > self.min_max_insert[1] and min_d < self.min_max_insert[0]:
    #!        return False
    #!    return True


class DeletionMove(BaseMove):
    """Class for performing a deletion move."""
    def __init__(self,
                 species: list[str],
                 seed : int,
                 operating_box : list[list] = None,
                 z_shift : float = None):
        super().__init__(species, seed)
        self.box = operating_box
        self.z_shift = z_shift

    def do_trial_move(self, atoms) -> int:
        """
        Delete a random atom from the structure.

        Returns:
        Atoms: Updated ASE Atoms object after the deletion.
        """
        trials = True
        atoms_new = atoms.copy()
        selected_species = self.rng.random.choice(self.species)
        indices_of_species = [atom.index for atom in atoms_new if atom.symbol in selected_species]
        if len(indices_of_species) == 0:
            return False, -1, 'X'
        if not self.z_shift:
            remove_index = self.rng.random.choice(indices_of_species)
            del atoms_new[remove_index]
            return atoms_new, -1, selected_species
        while trials:
            remove_index = self.rng.random.choice(indices_of_species)
            position = atoms_new[remove_index].position - np.array([0, 0, self.z_shift])
            if position[2] >= 0 and position[2] <= self.box[2][2]:
                trials = False
        del atoms_new[remove_index]
        return atoms_new, -1, selected_species