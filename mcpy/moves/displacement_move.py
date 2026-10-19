import numpy as np

from .base_move import BaseMove, fixed_mask
from ..cell import NullCell


class DisplacementMove(BaseMove):
    """Class for performing a displacement move.

    Mutates atom positions in place. The ensemble rolls back via the
    pre-trial arrays snapshot if the move is rejected.

    Candidates are the atoms whose symbol is in ``species``, minus the atoms
    fixed by a ``FixAtoms`` in ``atoms.constraints``, minus the indices in
    ``constraints``. The new positions go through ``atoms.set_positions``, so
    any other ASE constraint is applied too.

    ``constraints`` is a static list of atom *indices* excluded from
    displacement. GCMC insertions append at the end and deletions renumber
    everything after the removed index, so the convention is: constrained
    atoms (e.g. the support) must come FIRST in the Atoms object, before any
    GCMC-exchangeable species. Then deletions only ever touch higher indices
    and the constrained indices stay valid for the whole run. A ``FixAtoms``
    on the atoms does not have this limit: ASE remaps it on deletion.
    """

    def __init__(self,
                 species: list[str],
                 seed: int,
                 constraints: list = None,
                 max_displacement: float = 0.1,
                 n_steps: int = 1,
                 ) -> None:
        if n_steps < 1:
            raise ValueError(f'n_steps must be >= 1, got {n_steps}')
        cell = NullCell()
        super().__init__(cell, species, seed)
        self.max_displacement = max_displacement
        self.constraints = (np.empty(0, int) if constraints is None
                            else np.asarray(constraints, dtype=int).ravel())
        self.n_steps = int(n_steps)

    def _movable_indices(self, atoms):
        """Indices of ``species`` atoms that no constraint holds fixed."""
        n = len(atoms)
        mask = np.isin(atoms.get_chemical_symbols(), self.species)
        in_range = self.constraints[(self.constraints >= 0) & (self.constraints < n)]
        mask[in_range] = False
        mask &= ~fixed_mask(atoms)
        return np.flatnonzero(mask)

    def do_trial_move(self, atoms):
        """
        Displace ``self.n_steps`` distinct random atoms by a random vector
        within ``max_displacement``. With ``n_steps=1`` (default) this is the
        classic single-atom displacement; larger values bundle several
        displacements into a single trial for basin-hopping-style sampling.
        With fewer movable atoms than ``n_steps``, all of them move. This
        stays symmetric: a displacement does not change the movable count.
        """
        to_move = self._movable_indices(atoms)
        if to_move.size == 0:
            # An empty GCMC box, or no atom of ``species`` present yet: a
            # failed proposal, like a deletion with nothing to delete.
            return False, 0, 'X'
        k = min(self.n_steps, to_move.size)
        if self.n_steps == 1:
            selected = (int(self.rng.random.choice(to_move)),)
        else:
            selected = tuple(int(i) for i in self.rng.random.sample(list(to_move), k))

        positions = atoms.get_positions()
        for atom_index in selected:
            rsq = 1.1
            while rsq > 1.0:
                rx = 2.0 * self.rng.get_uniform() - 1.0
                ry = 2.0 * self.rng.get_uniform() - 1.0
                rz = 2.0 * self.rng.get_uniform() - 1.0
                rsq = rx * rx + ry * ry + rz * rz
            positions[atom_index] += np.array([rx, ry, rz]) * self.max_displacement
        atoms.set_positions(positions)
        return atoms, 0, 'X'
