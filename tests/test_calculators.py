"""BaseCalculator, MACE_F_Calculator and the lazy calculator imports.

No test here imports torch: the package import runs in a subprocess, and
``MACE_F_Calculator`` is checked against a stub ``mace.calculators`` module.
"""
import importlib
import logging
import os
import pathlib
import subprocess
import sys
import types

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones

from mcpy.calculators import BaseCalculator
from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector


REPO = pathlib.Path(__file__).resolve().parents[1]


def test_calculators_import_does_not_load_torch():
    code = (
        'import sys, mcpy.calculators as c\n'
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "names = {'BaseCalculator', 'MACE_F_Calculator', 'AlchemiCalculator',\n"
        "         'AlchemiFCalculator'}\n"
        'assert set(c.__all__) == names, c.__all__\n'
        'assert names <= set(dir(c)), dir(c)\n'
    )
    env = dict(os.environ, PYTHONPATH=str(REPO))
    proc = subprocess.run([sys.executable, '-c', code], cwd=REPO, env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_unknown_calculator_name_is_attribute_error():
    import mcpy.calculators as c
    with pytest.raises(AttributeError):
        c.NoSuchCalculator


def _rattled_cu():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.1, seed=3)
    return atoms


def test_base_calculator_steps0_is_single_point(caplog):
    atoms = _rattled_cu()
    start = atoms.positions.copy()
    ref = atoms.copy()
    ref.calc = EMT()
    calc = BaseCalculator(EMT(), steps=0, fmax=0.05)
    with caplog.at_level(logging.WARNING, logger='mcpy.calculators'):
        e = calc.get_potential_energy(atoms)
    assert not caplog.records
    assert e == ref.get_potential_energy()
    assert np.array_equal(atoms.positions, start)
    assert calc.last_relax_steps == 0


@pytest.fixture
def stub_mace(monkeypatch):
    """Replace ``mace.calculators`` with a recording stub (no torch)."""
    class StubMACECalculator(EMT):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

    mace = types.ModuleType('mace')
    mace_calcs = types.ModuleType('mace.calculators')
    mace_calcs.MACECalculator = StubMACECalculator
    mace.calculators = mace_calcs
    monkeypatch.setitem(sys.modules, 'mace', mace)
    monkeypatch.setitem(sys.modules, 'mace.calculators', mace_calcs)
    monkeypatch.delitem(sys.modules, 'mcpy.calculators.mace_f_calculator', raising=False)
    module = importlib.import_module('mcpy.calculators.mace_f_calculator')
    yield module, StubMACECalculator
    sys.modules.pop('mcpy.calculators.mace_f_calculator', None)


def test_mace_f_passes_head_and_dtype(stub_mace):
    module, _ = stub_mace
    calc = module.MACE_F_Calculator('x.model', steps=5, fmax=0.05,
                                    head='omat_pbe', default_dtype='float64')
    assert calc.calculator.kwargs['head'] == 'omat_pbe'
    assert calc.calculator.kwargs['default_dtype'] == 'float64'


def test_mace_f_rejects_head_on_prebuilt_calculator(stub_mace):
    module, stub = stub_mace
    with pytest.raises(ValueError, match='head'):
        module.MACE_F_Calculator(stub(), steps=5, fmax=0.05, head='omat_pbe')


def test_mace_f_warns_on_step_cap(stub_mace, caplog):
    module, _ = stub_mace
    calc = module.MACE_F_Calculator('x.model', steps=1, fmax=1e-6, optimizer='fire')
    with caplog.at_level(logging.WARNING, logger='mcpy.calculators'):
        calc.get_potential_energy(_rattled_cu())
    assert any('step cap' in r.getMessage() for r in caplog.records)
    assert calc.total_relax_steps == 1


def _lj_relax():
    return BaseCalculator(LennardJones(sigma=1.0, epsilon=1.0, rc=3.0),
                          steps=50, fmax=0.05)


def test_base_calculator_empty_structure_is_zero():
    calc = _lj_relax()
    calc.last_relax_steps = 7
    assert calc.get_potential_energy(Atoms(cell=[9, 9, 9], pbc=True)) == 0.0
    assert calc.last_relax_steps == 0


@pytest.mark.parametrize('start, mu', [
    (0, 5.0),     # GCMC from an empty box
    (1, -40.0),   # delete the last atom
])
def test_gcmc_with_relaxing_calculator_handles_empty_box(tmp_path, start, mu):
    atoms = Atoms('H' * start, positions=[[4.5, 4.5, 4.5]] * start,
                  cell=[9, 9, 9], pbc=True)
    cell = Cell(atoms, seed=1)
    moves = [InsertionMove(cell, species=['H'], seed=2),
             DeletionMove(cell, species=['H'], seed=3)]
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], calculator=_lj_relax(), mu={'H': mu},
        species=['H'], units_type='LJ', temperature=2.0,
        move_selector=MoveSelector([1, 1], moves, seed=4),
        random_seed=5, outfile=str(tmp_path / 'o.out'),
        traj_file=str(tmp_path / 't.xyz'))
    gcmc.run(10)
    assert (len(gcmc.atoms) > 0) == (start == 0)
