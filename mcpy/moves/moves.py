from abc import ABC, abstractmethod
from ase import Atoms
from ..utils import RandomNumberGenerator
import numpy as np
from ase.geometry import wrap_positions
from sklearn.metrics import pairwise_distances

class BaseMove(ABC):
    """Abstract base class for Monte Carlo moves."""

    def __init__(self, species: list[str], seed: int) -> None:
        """
        Initializes the move with the given atomic configuration, species, and RNG.

        Parameters:
        atoms (Atoms): ASE Atoms object representing the system.
        species (list[str]): List of possible atomic species for insertion.
        rng (RandomNumberGenerator): Random number generator.
        """
        self.species = species
        self.rng = RandomNumberGenerator(seed=seed)

    @abstractmethod
    def do_trial_move(self, atoms) -> Atoms:
        """
        Perform the Monte Carlo move and return the new atomic configuration.

        Returns:
        Atoms: Updated ASE Atoms object after the move.
        """
        pass

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
        #!excluded_volume = (4/3) * np.pi * (self.min_insert**3) * len(positions_bias)
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
            if min_dist >= self.min_insert:
                configurational_bias = True                    
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




class InsertionMove(BaseMove):
    """Class for performing an insertion move."""
    def __init__(self,
                 species: list[str],
                 seed : int,
                 operating_box : list[list] = None,
                 z_shift : float = None,
                 min_max_insert : list[float] = None):
        super().__init__(species, seed)
        self.box = operating_box
        self.z_shift = z_shift
        self.min_max_insert = min_max_insert

    def do_trial_move(self, atoms) -> Atoms:
        """
        Insert a random atom of a random species at a random position.

        Returns:
        Atoms: Updated ASE Atoms object after the insertion.
        """
        atoms_new = atoms.copy()
        selected_species = self.rng.random.choice(self.species)
        position = np.array([
            self.box[i]*self.rng.get_uniform() for i in range(3)
            ]).sum(axis=0)
        if self.z_shift:
            position[2] += self.z_shift
        atoms_new += Atoms(selected_species, positions=[position])
        if self.min_max_insert:
            if self.check_distance_criteria(atoms_new):
                return atoms_new, 1, selected_species
            else:
                return False, False, False
        return atoms_new, 1, selected_species

    def check_distance_criteria(self, atoms_new):
        min_d = min(atoms_new.get_distances(-1, range(len(atoms_new)-1), mic=True))
        if min_d > self.min_max_insert[1] and min_d < self.min_max_insert[0]:
            return False
        return True


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
        super().__init__(species, seed)
        self.max_displacement = max_displacement
        if len(constraints) > 0:
            self.constraints = constraints[0].todict()['kwargs']['indices']
        else:
            self.constraints = []

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
        atoms_new.set_positions(wrap_positions(atoms_new.positions, atoms_new.cell))
        return atoms_new, 0, 'X'


class PermutationMove(BaseMove):
    """Class for performing a permutation move.

    Returns:
        Atoms: Updated ASE Atoms object after the permutation."""
    def __init__(self,
                 species: list[str],
                 seed: int
                 ) -> None:
        """
        Initializes the permutation move.

        """
        super().__init__(species, seed)

    def do_trial_move(self, atoms) -> Atoms:
        """
        Permute the symbols of two random atoms.

        Returns:
        Atoms: Updated ASE Atoms object after the permutation.
        """
        atoms_new = atoms.copy()
        indices_symbol_a = [atom.index for atom in atoms_new if atom.symbol == self.species[0]]
        indices_symbol_b = [atom.index for atom in atoms_new if atom.symbol == self.species[1]]
        if len(indices_symbol_a) == 0 or len(indices_symbol_b) == 0:
            return False, 0, 'X'
        i = self.rng.random.choice(indices_symbol_a)
        j = self.rng.random.choice(indices_symbol_b)
        atoms_new[i].symbol, atoms_new[j].symbol = \
            atoms_new[j].symbol, atoms_new[i].symbol
        return atoms_new, 0, 'X'
