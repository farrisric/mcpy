"""Old min_insert rule vs the ADR 0003 free-volume rule, on O/Ag(111).

Three arms run the same GCMC (O insertion + deletion, relaxed trials):

  old      pre-ADR-0003 behaviour: insertions redraw only within MIN_INSERT
           of an atom, but the acceptance uses the free volume of the
           species_radii spheres; deletions use the pre-deletion volume.
  new      ADR 0003 with the calibrated radii (Ag 2.14, O 0.5): proposal and
           acceptance share one exclusion rule.
  new_small ADR 0003 with every radius at MIN_INSERT: the old "let the
           relaxation sort it out" idea, made consistent.

Per arm it records the cheap stage-1 redraws, the expensive relaxed trials,
their FIRE steps and outcome, and the O coverage.

    python benchmark/insertion_rule_comparison.py <arm> <out.json>
"""
import json
import sys
import time

import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.geometry import find_mic

from mcpy.calculators import AlchemiFCalculator
from mcpy.cell import CustomCell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector
from mcpy.utils import derive_mu_gas

CHECKPOINT = 'medium-mpa-0'
RELAX_STEPS = 300
FMAX = 0.05
AG_LATTICE = 4.165
R_O_AG = 2.14
R_O = 0.5
MIN_INSERT = 0.5
SLAB_SIZE = (6, 6, 4)
CELL_HEIGHT = 7.0
TEMPERATURE = 500.0
DELTA_MU_O = -0.3
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
SEED = 42
MAX_ATTEMPTS = 1000


class OldInsertionMove(InsertionMove):
    """InsertionMove as it was before 43f381b: uniform region point,
    redrawn only within ``min_insert`` of a cell-species atom."""

    def __init__(self, cell, species, seed, min_insert):
        super().__init__(cell, species, seed)
        self._min_sq = min_insert ** 2

    def do_trial_move(self, atoms):
        from ase import Atoms
        sp = self.rng.random.choice(self.species)
        pos = self.cell.get_random_point()
        idx = self.cell.get_atoms_specie_inside_cell(atoms, self.cell.get_species())
        if len(idx) > 0:
            ref = atoms.positions[idx]
            for _ in range(MAX_ATTEMPTS):
                diff = find_mic(ref - pos, atoms.cell, pbc=atoms.pbc)[0]
                if np.einsum('ij,ij->i', diff, diff).min() >= self._min_sq:
                    break
                pos = self.cell.get_random_point()
            else:
                return False, 1, sp
        atoms += Atoms(sp, positions=[pos])
        return atoms, 1, sp


class OldDeletionMove(DeletionMove):
    """Pre-ADR-0003 deletion: volume of the state still holding the atom."""

    def get_volume(self):
        return self.cell.get_volume()


def build_slab():
    nx, ny, nl = SLAB_SIZE
    atoms = fcc111('Ag', a=AG_LATTICE, size=(nx, ny, nl), periodic=True, vacuum=10.0)
    atoms.set_constraint(FixAtoms(indices=[a.index for a in atoms if a.tag >= nl - 1]))
    return atoms


def main():
    arm, out = sys.argv[1], sys.argv[2]
    s = [int(x) for x in np.random.SeedSequence(SEED).generate_state(5, dtype=np.uint32)]
    atoms = build_slab()
    z_top = float(atoms.positions[atoms.get_tags() == 1, 2].max())
    radii = ({'Ag': MIN_INSERT, 'O': MIN_INSERT} if arm == 'new_small'
             else {'Ag': R_O_AG, 'O': R_O})
    cell = CustomCell(atoms, custom_height=CELL_HEIGHT, bottom_z=z_top - R_O_AG,
                      species_radii=radii, seed=s[0])

    draws = [0]
    raw_point = cell.get_random_point

    def counted_point():
        draws[0] += 1
        return raw_point()
    cell.get_random_point = counted_point

    if arm == 'old':
        moves = [OldInsertionMove(cell, ['O'], s[1], MIN_INSERT),
                 OldDeletionMove(cell, ['O'], s[2])]
    else:
        moves = [InsertionMove(cell, ['O'], s[1]), DeletionMove(cell, ['O'], s[2])]

    calc = AlchemiFCalculator(checkpoint=CHECKPOINT, steps=RELAX_STEPS, fmax=FMAX,
                              device='cuda', enable_cueq=True, compile_model=True,
                              optimizer='fire2')
    mu_o = derive_mu_gas(calc, 'O2') + DELTA_MU_O
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], calculator=calc, mu={'O': mu_o}, species=['O'],
        units_type='metal', temperature=TEMPERATURE,
        move_selector=MoveSelector([1, 1], moves, seed=s[3]), random_seed=s[4],
        outfile=f'benchmark/insertion_rule_{arm}.out', traj_file=None,
        outfile_write_interval=50)

    trials = []
    raw_commit = gcmc._commit_or_rollback
    raw_propose = gcmc._propose

    def propose():
        before = draws[0]
        p = raw_propose()
        ins = gcmc.move_selector.to_use == 0
        trials.append({'move': 'ins' if ins else 'del',
                       'draws': draws[0] - before, 'proposed': p is not None,
                       'n_o': gcmc.atoms.get_chemical_symbols().count('O')})
        return p

    def commit(E_new, snapshot, meta):
        acc = raw_commit(E_new, snapshot, meta)
        trials[-1].update(accepted=bool(acc), fire=int(calc.last_relax_steps),
                          n_o=gcmc.atoms.get_chemical_symbols().count('O'))
        return acc

    gcmc._propose = propose
    gcmc._commit_or_rollback = commit

    t0 = time.perf_counter()
    gcmc.run(STEPS)
    wall = time.perf_counter() - t0
    with open(out, 'w') as fh:
        json.dump({'arm': arm, 'mu_o': mu_o, 'wall': wall,
                   'trials': trials, 'volume_empty': cell.free_volume(build_slab())},
                  fh)


if __name__ == '__main__':
    main()
