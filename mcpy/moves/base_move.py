from abc import ABC, abstractmethod
from ase import Atoms
from ..utils import RandomNumberGenerator
from ..cell import Cell

# Returned by a move in place of the atoms when it proposed a configuration
# whose acceptance probability is zero by construction (e.g. a molecule
# displacement that leaves the region). ``MoveSelector`` counts it as a
# rejected attempt, not a failed proposal, and hands the ensembles the usual
# ``False`` so no energy is evaluated.
PROPOSAL_REJECTED = object()


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
