from abc import ABC, abstractmethod

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms

from ..utils import RandomNumberGenerator
from ..cell import Cell

# Returned by a move in place of the atoms when it proposed a configuration
# whose acceptance probability is zero by construction (e.g. a molecule
# displacement that leaves the region). ``MoveSelector`` counts it as a
# rejected attempt, not a failed proposal, and hands the ensembles the usual
# ``False`` so no energy is evaluated.
PROPOSAL_REJECTED = object()


def fixed_mask(atoms):
    """Per-atom boolean: is this atom held by a ``FixAtoms``?"""
    mask = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            mask[constraint.get_indices()] = True
    return mask


def free_atom_count(atoms, species):
    """Atoms of ``species`` outside molecules (``molecule_id < 0``)."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    mol_ids = atoms.arrays.get('molecule_id')
    if mol_ids is not None:
        symbols = symbols[mol_ids < 0]
    return int(np.count_nonzero(symbols == species))


class BaseMove(ABC):
    """Abstract base class for Monte Carlo moves."""

    def __init__(self, cell: Cell, species: list[str], seed: int) -> None:
        """
        Initializes the move with its cell, species, and RNG.

        Parameters:
        cell (Cell): Region the move acts in (``NullCell`` for moves that
        sample no region).
        species (list[str]): Species the move acts on.
        seed (int): Seed of the move's ``RandomNumberGenerator``. ``None``
        seeds it from OS entropy, so the run is not repeatable.
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
