from collections import Counter

import numpy as np
import pytest
from ase.cluster import Octahedron
from ase.calculators.emt import EMT
from ase.optimize import LBFGS
from ase.units import kB

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


def test_unproposable_move_is_a_non_accepted_step(tmp_path):
    # A single-species cluster: PermutationMove can never pick two species,
    # so do_trial_move returns False and trial_step must treat it as a
    # non-accepted step rather than crashing.
    atoms = Octahedron('Au', 2)  # all Au
    ms = MoveSelector([1.0], [PermutationMove(species=['Au', 'Pt'], seed=3)])
    mc = CanonicalEnsemble(
        atoms=atoms, calculator=EMT(), optimizer=LBFGS, fmax=0.2,
        temperature=600.0, move_selector=ms, random_seed=3,
        traj_file=str(tmp_path / 't.xyz'), outfile=str(tmp_path / 'o.out'),
    )
    mc.run(steps=4)
    assert mc._accepted_trials == 0
    assert Counter(mc.atoms.get_chemical_symbols()) == Counter({'Au': len(atoms)})
    assert np.isfinite(mc._current_energy)


def test_get_set_state_roundtrip(tmp_path):
    mc = _make_ensemble(tmp_path, temperature=600.0, seed=2)
    mc.initialize_run()  # relaxes initial config, sets _current_energy
    try:
        state = mc.get_state()
        assert state['beta'] == pytest.approx(1.0 / (kB * 600.0))
        assert 'energy' in state and 'atoms' in state
        assert 'mu' not in state

        saved_energy = state['energy']
        mc._current_energy = 999.0
        mc.atoms = _make_cluster()  # a different Atoms object

        mc.set_state(state)
        assert mc._current_energy == saved_energy
        kvp = mc.atoms.info['key_value_pairs']['potential_energy']
        assert kvp == saved_energy
    finally:
        mc.finalize_run()
