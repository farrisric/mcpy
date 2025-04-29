from ase import Atoms

from .base_move import BaseMove


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
        cell = None
        super().__init__(cell, species, seed)

    def do_trial_move(self, atoms) -> Atoms:
        """
        Permute the symbols of two random atoms.

        Returns:
        Atoms: Updated ASE Atoms object after the permutation.
        """
        atoms_new = atoms.copy()
        species_pair = self.rng.random.choice(self.species, 2, replace=False)
        indices_symbol_a = [atom.index for atom in atoms_new if atom.symbol == species_pair[0]]
        indices_symbol_b = [atom.index for atom in atoms_new if atom.symbol == species_pair[1]]
        if len(indices_symbol_a) == 0 or len(indices_symbol_b) == 0:
            return False, 0, 'X'
        i = self.rng.random.choice(indices_symbol_a)
        j = self.rng.random.choice(indices_symbol_b)
        atoms_new[i].symbol, atoms_new[j].symbol = \
            atoms_new[j].symbol, atoms_new[i].symbol
        return atoms_new, 0, 'X'
