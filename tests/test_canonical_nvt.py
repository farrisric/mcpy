import logging
import random
from collections import Counter

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.cluster import Octahedron
from ase.constraints import FixAtoms
from ase.optimize import BFGS, LBFGS
from ase.units import kB

from mcpy.ensembles.canonical_ensemble import CanonicalEnsemble
from mcpy.moves import DisplacementMove, ShakeMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.moves.permutation_move import PermutationMove

from helpers import _canonical


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


def _count_xyz_frames(path):
    """Count frames in an extended-xyz file. A frame begins with a line
    containing only the atom count."""
    n_frames = 0
    with open(path) as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            try:
                n_atoms = int(header.strip())
            except ValueError:
                continue
            n_frames += 1
            # comment line + n_atoms coordinate lines
            for _ in range(n_atoms + 1):
                fh.readline()
    return n_frames


def test_minima_file_append_only_records_improvements(tmp_path):
    ms = MoveSelector([1.0], [PermutationMove(species=['Au', 'Pt'], seed=7)])
    mc = CanonicalEnsemble(
        atoms=_make_cluster(),
        calculator=EMT(), optimizer=LBFGS, fmax=0.2,
        temperature=600.0, move_selector=ms, random_seed=7,
        traj_file=str(tmp_path / 't.xyz'),
        outfile=str(tmp_path / 'o.out'),
        minima_file=str(tmp_path / 'minima.xyz'),
        minima_mode='a',
    )
    mc.run(steps=6)
    minima_path = tmp_path / 'minima.xyz'
    assert minima_path.exists()
    n_frames = _count_xyz_frames(minima_path)
    # initialize_run records the relaxed initial config as the first minimum;
    # any further frames must each correspond to a strict improvement.
    assert n_frames >= 1
    assert mc._best_atoms is not None
    assert mc._best_score == pytest.approx(mc._best_energy)


def test_minima_file_overwrite_keeps_single_frame(tmp_path):
    ms = MoveSelector([1.0], [PermutationMove(species=['Au', 'Pt'], seed=11)])
    mc = CanonicalEnsemble(
        atoms=_make_cluster(),
        calculator=EMT(), optimizer=LBFGS, fmax=0.2,
        temperature=600.0, move_selector=ms, random_seed=11,
        traj_file=None,
        outfile=str(tmp_path / 'o.out'),
        minima_file=str(tmp_path / 'best.xyz'),
        minima_mode='w',
    )
    mc.run(steps=6)
    best_path = tmp_path / 'best.xyz'
    assert best_path.exists()
    assert _count_xyz_frames(best_path) == 1


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
    finally:
        mc.finalize_run()


def _canon(**kw):
    from ase.optimize import LBFGS
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    kw.setdefault('optimizer', LBFGS)
    return CanonicalEnsemble(atoms=atoms, calculator=EMT(),
                             move_selector=MoveSelector([1], [DisplacementMove(
                                 species=['Cu'], seed=1)]),
                             random_seed=1, traj_file=None, outfile=None, **kw)


# CanonicalEnsemble: T = 0 and the relaxation step cap

def test_canonical_zero_temperature_is_greedy():
    mc = _canon(temperature=0)
    assert mc._acceptance_condition(-0.1) is True
    assert not mc._acceptance_condition(1e-9)


def test_canonical_relax_is_step_capped(caplog):
    calls = {}

    class _NeverConverges:
        def __init__(self, atoms, logfile=None):
            pass

        def run(self, fmax, steps=None):
            calls['steps'] = steps
            return False

    mc = _canon(optimizer=_NeverConverges, steps=7)
    with caplog.at_level(logging.WARNING, logger='mcpy'):
        mc.relax(mc.atoms)
    assert calls['steps'] == 7
    assert 'did not converge' in caplog.text


# CanonicalEnsemble hygiene (bugs: seeded/drew from the global random module;
# trajectory_write_interval accepted but ignored)

def test_canonical_does_not_touch_global_random(tmp_path):
    random.seed(123456)
    state_before = random.getstate()
    mc = _canonical(tmp_path)
    assert random.getstate() == state_before
    mc.run(steps=2)
    assert random.getstate() == state_before


def test_canonical_honors_trajectory_write_interval(tmp_path):
    mc = _canonical(tmp_path, trajectory_write_interval=2)
    mc.run(steps=4)
    # Initial frame + one frame at each interval boundary (steps 2 and 4),
    # independent of how many trials were accepted.
    assert _count_xyz_frames(tmp_path / 't.xyz') == 3


def _cu_cluster():
    return Atoms('Cu4', positions=[[0, 0, 0], [2.3, 0.2, 0], [0, 2.6, 0.1], [0.3, 0, 2.2]],
                 cell=[12] * 3, pbc=False)


# Canonical constraints, failed setup, final row

def test_canonical_constraints_hold_during_the_initial_relaxation(tmp_path):
    atoms = _cu_cluster()
    before = atoms.positions[0].copy()
    ens = CanonicalEnsemble(atoms, EMT(), BFGS, MoveSelector([1], [ShakeMove(0.1, 1)]),
                            constraints=FixAtoms([0]), random_seed=1, steps=50,
                            traj_file=None, outfile=None)
    ens.run(2)
    np.testing.assert_allclose(ens.atoms.positions[0], before)


def test_canonical_writes_the_last_row_and_frame(tmp_path):
    out, traj = tmp_path / 'c.out', tmp_path / 'c.xyz'
    ens = CanonicalEnsemble(_cu_cluster(), EMT(), BFGS,
                            MoveSelector([1], [ShakeMove(0.1, 1)]), random_seed=1,
                            steps=20, outfile=str(out), traj_file=str(traj),
                            outfile_write_interval=10, trajectory_write_interval=10)
    ens.run(3)
    ens.run(3)
    steps = [line.split()[1] for line in open(out) if line.startswith('STEP:')]
    assert steps == ['0', '3', '6']
    assert len(ase.io.read(str(traj), index=':')) == 3


# Input validation before the first energy

def test_canonical_rejects_negative_temperature():
    with pytest.raises(ValueError, match='temperature'):
        CanonicalEnsemble(None, None, None, None, temperature=-1.0,
                          traj_file=None, outfile=None)
