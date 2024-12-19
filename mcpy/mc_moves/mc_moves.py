from abc import ABC, abstractmethod
from ase import Atoms
from npl.utils import RandomNumberGenerator
import numpy as np
from ase.geometry import wrap_positions


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
                 seed : int,
                 operating_box : list[list] = None,
                 z_shift : float = None):
        super().__init__(species, seed)
        self.box = operating_box
        self.z_shift = z_shift

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
        return atoms_new, 1, selected_species


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
                 constraints: list,
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
        displacement = [
            self.rng.get_uniform(-self.max_displacement, self.max_displacement) for _ in range(3)
            ]
        atoms_new.positions[atom_index] += displacement
        atoms_new.set_positions(wrap_positions(atoms_new.positions, atoms_new.cell))
        return atoms_new
