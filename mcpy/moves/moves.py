from abc import ABC, abstractmethod
from ase import Atoms
from ..utils import RandomNumberGenerator
from ..cell import Cell
import numpy as np
from sklearn.metrics import pairwise_distances


class BaseMove(ABC):
    """Abstract base class for Monte Carlo moves."""

    def __init__(self, cell: Cell, species: list[str], seed: int) -> None:
        """
        Initializes the move with the given atomic configuration, species, and RNG.

        Parameters:
        atoms (Atoms): ASE Atoms object representing the system.
        species (list[str]): List of possible atomic species for insertion.
        rng (RandomNumberGenerator): Random number generator.
        """
        self.species = species
        self.cell = cell
        self.rng = RandomNumberGenerator(seed=seed)

    @abstractmethod
    def do_trial_move(self, atoms) -> Atoms:
        """
        Perform the Monte Carlo move and return the new atomic configuration.

        Returns:
        Atoms: Updated ASE Atoms object after the move.
        """
        pass

    def get_volume(self) -> float:
        """
        Calculate the volume of the cell.

        Returns:
        float: Volume of the cell.
        """
        return self.cell.get_volume()

    def calculate_volume(self, atoms) -> None:
        """
        Update the volume of the cell.

        Returns:
        None
        """
        self.cell.calculate_volume(atoms)


class InsertionMove(BaseMove):
    """Class for performing an insertion move."""
    def __init__(self,
                 cell : Cell,
                 species: list[str],
                 seed : int,
                 min_insert : float = None,
                 max_insert : float = None,
                 species_bias : list[str] = None) -> None:
        super().__init__(cell, species, seed)
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
        positions_bias = atoms_new[self.cell.get_atoms_specie_inside_cell(
            atoms_new, self.species_bias)].positions

        insert_position = self.cell.get_random_point()
        min_dist = np.min(pairwise_distances(
            insert_position.reshape(1, -1), positions_bias).flatten())

        while min_dist < self.min_insert:
            insert_position = self.cell.get_random_point()
            min_dist = np.min(pairwise_distances(
                insert_position.reshape(1, -1), positions_bias).flatten())
        atoms_new += Atoms(selected_species, positions=[insert_position])
        return atoms_new, 1, selected_species


class DeletionMove(BaseMove):
    """Class for performing a deletion move."""
    def __init__(self,
                 cell : Cell,
                 species: list[str],
                 seed : int,):
        super().__init__(cell, species, seed)

    def do_trial_move(self, atoms) -> int:
        """
        Delete a random atom from the structure.

        Returns:
        Atoms: Updated ASE Atoms object after the deletion.
        """
        atoms_new = atoms.copy()
        selected_species = self.rng.random.choice(self.species)
        indices_of_species = self.cell.get_atoms_specie_inside_cell(
            atoms_new, selected_species)
        if len(indices_of_species) == 0:
            return False, -1, 'X'
        remove_index = self.rng.random.choice(indices_of_species)
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
