from collections import Counter

import numpy as np
from ase.cluster import Octahedron
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from mcpy.ensembles.canonical_ensemble import CanonicalEnsemble
from mcpy.moves.move_selector import MoveSelector
from mcpy.moves.permutation_move import PermutationMove


def _make_cluster():
    atoms = Octahedron('Au', 2)  # 6-atom regular octahedron, fast to relax
    half = len(atoms) // 2
    atoms.symbols = ['Au'] * half + ['Pt'] * (len(atoms) - half)
    return atoms


def _make_ensemble(tmp_path, temperature=600.0, seed=1):
    ms = MoveSelector([1.0], [PermutationMove(species=['Au', 'Pt'], seed=seed)])
    return CanonicalEnsemble(
        atoms=_make_cluster(),
        calculator=EMT(),
        optimizer=LBFGS,
        fmax=0.2,
        temperature=temperature,
        move_selector=ms,
        random_seed=seed,
        traj_file=str(tmp_path / 't.xyz'),
        outfile=str(tmp_path / 'o.out'),
    )


def test_nvt_permutation_conserves_composition(tmp_path):
    mc = _make_ensemble(tmp_path)
    before = Counter(mc.atoms.get_chemical_symbols())
    mc.run(steps=5)
    after = Counter(mc.atoms.get_chemical_symbols())
    assert before == after
    assert np.isfinite(mc._current_energy)
