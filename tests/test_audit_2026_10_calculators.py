"""Regressions from the 2026-10 calculator audit.

The CPU tests never import torch: the package import runs in a subprocess, and
``MACE_F_Calculator`` is checked against a stub ``mace.calculators`` module.
The GPU tests are opt-in (``MCPY_GPU_TESTS=1``), for the reason given in
``tests/test_ag111_gcmc_golden.py``: a torch import probe can SIGBUS.
"""
import importlib
import logging
import os
import pathlib
import subprocess
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111
from ase.calculators.emt import EMT
from ase.cluster import Octahedron
from ase.constraints import FixAtoms, FixCartesian
from ase.io import read

from mcpy.calculators import BaseCalculator

REPO = pathlib.Path(__file__).resolve().parents[1]


# ------------------------------------------------------------ R1: lazy import

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


# ------------------------------------------------- R3: BaseCalculator relaxing

def _rattled_cu():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True).repeat(2)
    atoms.rattle(0.1, seed=3)
    return atoms


def test_base_calculator_relaxes_and_counts_steps():
    atoms = _rattled_cu()
    start = atoms.positions.copy()
    unrelaxed = atoms.copy()
    unrelaxed.calc = EMT()
    e0 = unrelaxed.get_potential_energy()

    calc = BaseCalculator(EMT(), steps=200, fmax=0.01)
    e = calc.get_potential_energy(atoms)

    assert e <= e0
    assert not np.allclose(atoms.positions, start)
    assert calc.last_relax_steps > 0
    first = calc.last_relax_steps
    calc.get_potential_energy(_rattled_cu())
    assert calc.total_relax_steps == first + calc.last_relax_steps


def test_base_calculator_warns_on_step_cap(caplog):
    calc = BaseCalculator(EMT(), steps=1, fmax=1e-6)
    with caplog.at_level(logging.WARNING, logger='mcpy.calculators'):
        calc.get_potential_energy(_rattled_cu())
    assert any('step cap' in r.getMessage() for r in caplog.records)
    assert calc.last_relax_steps == 1


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


# ------------------------------------------ U10: MACE_F_Calculator passthrough

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


# ------------------------------------------------------------- GPU (opt-in)

CHECKPOINT = pathlib.Path(os.environ.get(
    'MCPY_TEST_CHECKPOINT',
    pathlib.Path.home() / '.cache/mace/macesmalldensityagnesistressmodel'))
MH1 = pathlib.Path.home() / '.cache/mace/macemh1model'

gpu = pytest.mark.skipif(
    not os.environ.get('MCPY_GPU_TESTS') or not CHECKPOINT.exists(),
    reason='set MCPY_GPU_TESTS=1 and provide a MACE checkpoint (GPU required)')


@pytest.fixture(scope='module')
def alchemi():
    from mcpy.calculators import alchemi
    return alchemi


@pytest.fixture(scope='module')
def device(alchemi):
    # CUDA when there is one; the gated tests also pass on the CPU.
    return os.environ.get('MCPY_TEST_DEVICE') or (
        'cuda' if alchemi.torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope='module')
def energy_calc(alchemi, device):
    return alchemi.AlchemiCalculator(str(CHECKPOINT), device=device,
                                     enable_cueq=False, compile_model=False)


@pytest.fixture(scope='module')
def relax_calc(alchemi, device):
    return alchemi.AlchemiFCalculator(str(CHECKPOINT), steps=50, device=device,
                                      enable_cueq=False, compile_model=False)


def _ag_slab():
    atoms = fcc111('Ag', size=(2, 2, 3), a=4.165, vacuum=6.0)
    atoms.rattle(0.05, seed=1)
    return atoms


@gpu
@pytest.mark.parametrize('which', ['energy_calc', 'relax_calc'])
def test_empty_structure_is_zero_energy(which, request):
    calc = request.getfixturevalue(which)
    assert calc.get_potential_energy(Atoms()) == 0.0
    slab = _ag_slab()
    ref = calc.get_potential_energies([slab.copy()])[0]
    out = calc.get_potential_energies([Atoms(), slab.copy(), Atoms()])
    assert out[0] == 0.0 and out[2] == 0.0
    assert out[1] == pytest.approx(ref, abs=1e-3)


@gpu
def test_relax_steps0_is_pure_single_point(alchemi, device, energy_calc, caplog):
    calc = alchemi.AlchemiFCalculator(str(CHECKPOINT), steps=0, device=device, enable_cueq=False,
                                      compile_model=False)
    slab = _ag_slab()
    start = slab.positions.copy()
    with caplog.at_level(logging.WARNING, logger='mcpy.calculators'):
        e = calc.get_potential_energy(slab)
    assert not caplog.records
    assert np.array_equal(slab.positions, start)
    assert e == pytest.approx(energy_calc.get_potential_energy(slab.copy()), abs=1e-4)


@gpu
def test_unsupported_constraint_raises(relax_calc, energy_calc):
    slab = _ag_slab()
    slab.set_constraint([FixAtoms(indices=[0]), FixCartesian(1, mask=(1, 1, 0))])
    with pytest.raises(NotImplementedError, match='FixCartesian'):
        relax_calc.get_potential_energy(slab)
    with pytest.raises(NotImplementedError, match='FixCartesian'):
        relax_calc.run_md(slab, temperature=300.0, steps=2)
    # A single point does not move atoms, so constraints are irrelevant there.
    assert np.isfinite(energy_calc.get_potential_energy(slab))


@gpu
def test_energy_only_defaults(alchemi, device, energy_calc, relax_calc):
    assert energy_calc.energy_only is True
    assert 'forces' not in energy_calc.model.model_config.active_outputs
    with pytest.raises(ValueError, match='energy_only'):
        energy_calc.run_md(_ag_slab(), temperature=300.0, steps=2)
    shared = alchemi.AlchemiCalculator(relax_calc.model, device=device)
    assert shared.energy_only is False
    assert 'forces' in relax_calc.model.model_config.active_outputs


@gpu
def test_run_md_does_not_drift_the_center_of_mass(relax_calc):
    # Before the per-step COM removal, the Langevin noise shifted this
    # 79-atom cluster by about 0.17 A per 100-step run at 2000 K.
    atoms = Octahedron('Cu', 5, 1)
    atoms.center(vacuum=8.0)
    atoms.pbc = True
    com_before = atoms.get_center_of_mass()
    relax_calc.run_md(atoms, temperature=2000.0, steps=100, dt=2.0, seed=3)
    assert np.linalg.norm(atoms.get_center_of_mass() - com_before) < 0.02


@gpu
def test_alchemi_brownian_writes_every_md_frame(relax_calc, tmp_path):
    from mcpy.moves import AlchemiBrownianMove
    path = tmp_path / 'md.xyz'
    atoms = Octahedron('Cu', 5, 1)
    atoms.center(vacuum=8.0)
    atoms.pbc = True
    move = AlchemiBrownianMove(relax_calc, temperature=2000.0, steps=5, dt=2.0,
                               seed=1, trajectory_file=str(path))
    move.do_trial_move(atoms)
    move.do_trial_move(atoms)
    frames = read(path, index=':')
    assert [f.info['md_run'] for f in frames] == [1] * 6 + [2] * 6
    assert [f.info['md_step'] for f in frames] == list(range(6)) * 2
    # initialize_velocities rescales to the target T, so a wrong velocity
    # unit would show up here as a T off by a factor of ~100.
    assert frames[0].info['T_K'] == pytest.approx(2000.0, rel=0.05)
    assert frames[-1].info['time_fs'] == pytest.approx(10.0)
    np.testing.assert_allclose(frames[-1].positions, atoms.positions, atol=1e-4)
    assert np.isfinite(frames[-1].get_forces()).all()


def _fake_mace(edge, first_edge, flag=None, node='512x0e'):
    first = SimpleNamespace(node_feats_irreps=node)
    if first_edge is not None:
        first.edge_irreps = first_edge
    model = SimpleNamespace(edge_irreps=edge, interactions=[first])
    if flag is not None:
        model.use_edge_irreps_first = flag
    return model


@gpu
def test_cueq_rebuild_check_on_mocked_models(alchemi):
    ok = alchemi._cueq_rebuild_matches
    # mh-1: built with scalar edge irreps on block 0, but predates the flag.
    assert not ok(_fake_mace('128x0e+128x1o', '128x0e'))
    assert ok(_fake_mace('128x0e+128x1o', '128x0e', flag=True))
    assert ok(_fake_mace('128x0e+128x1o', '512x0e'))
    assert ok(_fake_mace(None, '512x0e'))
    assert ok(_fake_mace(None, None))  # older blocks have no edge_irreps


def _uses_cueq(calc):
    return any('cuequivariance' in type(m).__module__
               for m in calc.model.modules())


@gpu
@pytest.mark.skipif(not MH1.exists(), reason='mh-1 checkpoint not cached')
@pytest.mark.skipif(os.environ.get('MCPY_TEST_DEVICE', 'cuda') != 'cuda',
                    reason='cuEq needs CUDA')
def test_mh1_cueq_is_repaired_and_matches_e3nn(alchemi, caplog):
    """mh-1 predates use_edge_irreps_first; mcpy sets it so the cuEq rebuild
    keeps block 0's real width. Three fresh cuEq loads must use cuEq kernels
    and match e3nn exactly (the unrepaired rebuild gave load-varying energies)."""
    import torch
    kw = dict(device='cuda', dtype=torch.float64, compile_model=False,
              head='mp_pbe_refit_add')
    ag = bulk('Ag', 'fcc', a=4.1592)
    e3nn = alchemi.AlchemiCalculator(str(MH1), enable_cueq=False, **kw)
    ref = e3nn.get_potential_energy(ag.copy())
    assert ref == pytest.approx(-2.81710, abs=1e-4)
    for _ in range(3):
        with caplog.at_level(logging.WARNING, logger='mcpy.calculators'):
            cueq = alchemi.AlchemiCalculator(str(MH1), enable_cueq=True, **kw)
        assert _uses_cueq(cueq)
        assert cueq.get_potential_energy(ag.copy()) == pytest.approx(ref, abs=1e-6)
    assert not any('cuEq disabled' in r.getMessage() for r in caplog.records)


def _model_forces(calc, atoms):
    from mcpy.calculators.alchemi import _build_nl
    batch = calc._batch(atoms)
    _build_nl(batch, calc._nl_hook())
    return calc.model(batch)['forces'].detach().cpu().numpy()


@gpu
@pytest.mark.skipif(not MH1.exists(), reason='mh-1 checkpoint not cached')
@pytest.mark.skipif(os.environ.get('MCPY_TEST_DEVICE', 'cuda') != 'cuda',
                    reason='cuEq needs CUDA')
def test_mh1_cueq_with_compile_gives_e3nn_forces(alchemi, caplog):
    """With cuEq, inductor miscompiled mh-1's non-linear blocks: forces were
    off by up to 0.6 eV/A on this rattled Pt55, energies right. compile is
    now turned off for such a model, so the forces match e3nn."""
    pt = Octahedron('Pt', 5, cutoff=2, latticeconstant=3.92)
    pt.center(vacuum=8.0)
    pt.pbc = True
    pt.rattle(0.1, seed=0)
    kw = dict(device='cuda', head='omat_pbe', energy_only=False)
    ref = _model_forces(alchemi.AlchemiCalculator(
        str(MH1), enable_cueq=False, compile_model=False, **kw), pt)
    with caplog.at_level(logging.WARNING, logger='mcpy.calculators'):
        cueq = alchemi.AlchemiCalculator(str(MH1), enable_cueq=True,
                                         compile_model=True, **kw)
    assert _uses_cueq(cueq)
    assert any('compile_model disabled' in r.getMessage() for r in caplog.records)
    np.testing.assert_allclose(_model_forces(cueq, pt), ref, rtol=0, atol=1e-4)
