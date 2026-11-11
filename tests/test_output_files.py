"""Outfile, trajectory and minima files: repeated runs, the last row and
frame, headers.
"""
import io

import ase.io
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms

from mcpy.cell import Cell, NullCell
from mcpy.ensembles.base_ensemble import write_xyz
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector

from helpers import _frames, _gcmc, _h2, _lj_gcmc, _selector_gcmc


def _rows(path):
    return [int(line.split()[0]) for line in open(path)
            if line.split() and line.split()[0].isdigit()]


# Run() twice appends; no duplicated final frame

def test_second_run_appends_without_duplicates(tmp_path):
    g = _lj_gcmc(tmp_path=tmp_path)
    g.run(3)
    g.run(3)
    out = tmp_path / 'g.out'
    assert open(out).read().count('Grand Canonical Ensemble') == 1
    assert _rows(out) == list(range(7))
    assert _frames(tmp_path / 'g.xyz') == 7


@pytest.mark.parametrize('interval, frames', [(1, 4), (2, 3)])
def test_finalize_writes_the_last_frame_once(tmp_path, interval, frames):
    """Steps 0..3 at interval 1; steps 0, 2 and the final 3 at interval 2."""
    g = _lj_gcmc(tmp_path=tmp_path, trajectory_write_interval=interval)
    g.run(3)
    assert _frames(tmp_path / 'g.xyz') == frames


def test_minima_single_best_survives_second_run(tmp_path):
    g = _lj_gcmc(tmp_path=tmp_path, minima_file=str(tmp_path / 'm.xyz'),
                 minima_mode='w')
    g.run(2)
    g.run(1)
    assert _frames(tmp_path / 'm.xyz') == 1


def test_interval_ratios_reset_with_outfile_disabled():
    atoms = _h2()
    g = _selector_gcmc(atoms, [NullCell()], MoveSelector([1], [DisplacementMove(
        species=['H'], seed=5, max_displacement=0.1)]), {'H': 0.0})
    assert g._outfile is None
    g.do_gcmc_step()
    assert g.move_selector.move_counter == [1]
    g.write_outfile()
    assert g.move_selector.move_counter == [0]
    assert g.move_selector.move_counter_total == [1]  # cumulative untouched


# The trajectory restarts the same system

def test_write_xyz_round_trips_pbc_constraints_tags_and_molecule_id():
    atoms = Atoms('CuOO', positions=[[0, 0, 0], [1, 1, 1], [1, 1, 2.2]],
                  cell=[5, 5, 12], pbc=(True, True, False))
    atoms.set_tags([1, 0, 2])
    atoms.set_constraint(FixAtoms([0]))
    atoms.new_array('molecule_id', np.array([-1, 0, 0]))
    atoms.calc = EMT()  # its results must not leak into the frame
    buf = io.StringIO()
    write_xyz(atoms, -1.5, buf, extra={'step': 3, 'score': -2.0})
    back = ase.io.read(io.StringIO(buf.getvalue()), format='extxyz')
    assert list(back.pbc) == [True, True, False]
    assert [list(c.get_indices()) for c in back.constraints] == [[0]]
    np.testing.assert_array_equal(back.get_tags(), [1, 0, 2])
    np.testing.assert_array_equal(back.arrays['molecule_id'], [-1, 0, 0])
    assert back.get_potential_energy() == pytest.approx(-1.5)
    assert back.info['step'] == 3 and back.info['score'] == pytest.approx(-2.0)


# Canonical constraints, failed setup, final row

def test_failed_setup_does_not_break_a_retried_run(tmp_path):
    atoms = Atoms(cell=[5] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    out = tmp_path / 'o.out'
    g = _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
              outfile=str(out), traj_file=str(tmp_path / 't.xyz'))
    real = g.initialize_outfile
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise OSError('disk full')
        real()

    g.initialize_outfile = flaky
    with pytest.raises(OSError):
        g.run(2)
    g.run(2)
    text = out.read_text()
    assert 'Simulation Parameters' in text
    rows = [line.split()[0] for line in text.splitlines() if line[:1].isdigit()]
    assert rows == ['0', '1', '2']


def _minima_gcmc(tmp_path, **kw):
    atoms = Atoms(cell=[6] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    return _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
                 mu={'H': 3.0}, minima_file=str(tmp_path / 'min.xyz'),
                 outfile=str(tmp_path / 'o.out'), traj_file=str(tmp_path / 't.xyz'), **kw)


def test_minima_history_starts_fresh_and_labels_the_outfile_step(tmp_path):
    (tmp_path / 'min.xyz').write_text('stale\n')
    g = _minima_gcmc(tmp_path)
    g.run(5)
    frames = ase.io.read(str(tmp_path / 'min.xyz'), index=':')
    traj = ase.io.read(str(tmp_path / 't.xyz'), index=':')
    # Each minimum is a frame of the trajectory with the same step label.
    n_at_step = {i: len(f) for i, f in enumerate(traj)}
    assert frames[0].info['step'] == 0
    for f in frames:
        assert len(f) == n_at_step[f.info['step']]
    assert [f.info['step'] for f in frames] == sorted({f.info['step'] for f in frames})
    assert max(f.info['step'] for f in frames) >= 1


# The outfile records the code version

def test_outfile_header_records_the_mcpy_version(tmp_path):
    import mcpy
    atoms = Atoms(cell=[5] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    out = tmp_path / 'o.out'
    _gcmc(atoms, [cell], [InsertionMove(cell, ['H'], 2), DeletionMove(cell, ['H'], 3)],
          outfile=str(out)).run(1)
    assert f'mcpy version: {mcpy.__version__}' in out.read_text()
