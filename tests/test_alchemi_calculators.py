"""AlchemiCalculator and AlchemiFCalculator.

Opt-in: set ``MCPY_GPU_TESTS=1`` and provide a MACE checkpoint. The gate is
an environment variable and not an import probe, because a torch import
probe can SIGBUS (see ``tests/test_ag111_gcmc_golden.py``).
"""
import logging
import os
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, fcc111
from ase.cluster import Octahedron
from ase.constraints import FixAtoms, FixCartesian
from ase.io import read


CHECKPOINT = pathlib.Path(os.environ.get(
    'MCPY_TEST_CHECKPOINT',
    pathlib.Path.home() / '.cache/mace/macesmalldensityagnesistressmodel'))


gpu = pytest.mark.skipif(
    not os.environ.get('MCPY_GPU_TESTS') or not CHECKPOINT.exists(),
    reason='set MCPY_GPU_TESTS=1 and provide a MACE checkpoint (GPU required)')


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


MH1 = pathlib.Path.home() / '.cache/mace/macemh1model'


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


@pytest.fixture(scope='module')
def torch():
    import torch
    return torch


@pytest.fixture(scope='module')
def kw(torch, device):
    return dict(device=device, dtype=torch.float64, enable_cueq=False,
                compile_model=False)


@pytest.fixture(scope='module')
def energy_calc64(alchemi, kw):
    return alchemi.AlchemiCalculator(str(CHECKPOINT), **kw)


def _fixed_ag_slab(seed=1, fixed=True):
    atoms = fcc111('Ag', size=(2, 2, 3), a=4.165, vacuum=6.0)
    atoms.rattle(0.1, seed=seed)
    if fixed:
        atoms.set_constraint(FixAtoms(indices=range(4)))
    return atoms


# Not _ALCHEMI_OPTIMIZERS: collection must not import torch.
OPTIMIZERS = ['fire', 'fire2', 'lbfgs']


def _relax_calc(alchemi, kw, optimizer, **extra):
    if optimizer not in alchemi._ALCHEMI_OPTIMIZERS:
        pytest.skip(f'{optimizer} needs a newer nvalchemi')
    return alchemi.AlchemiFCalculator(str(CHECKPOINT), optimizer=optimizer,
                                      **dict(kw, **extra))


@gpu
@pytest.mark.parametrize('optimizer', OPTIMIZERS)
def test_relax_steps0_is_single_point(alchemi, kw, energy_calc64, optimizer, caplog):
    # nvalchemi rejects n_steps=0, so the single point is built before the
    # optimizer. No step-cap warning either: nothing was capped.
    calc = _relax_calc(alchemi, kw, optimizer, steps=0)
    slab = _fixed_ag_slab()
    start = slab.positions.copy()
    with caplog.at_level(logging.WARNING, logger='mcpy.calculators'):
        e = calc.get_potential_energy(slab)
    assert not caplog.records
    assert np.array_equal(slab.positions, start)
    assert calc.last_relax_steps == 0
    assert e == pytest.approx(energy_calc64.get_potential_energy(slab), abs=1e-8)


@gpu
@pytest.mark.parametrize('optimizer', OPTIMIZERS)
def test_batched_relax_matches_single_and_holds_fixed(alchemi, kw, optimizer, caplog):
    # calculators-lbfgs-untested-cuda-only-tests
    calc = _relax_calc(alchemi, kw, optimizer, steps=300)
    slabs = [_fixed_ag_slab(seed) for seed in (1, 2, 3)]
    fixed0 = [s.positions[:4].copy() for s in slabs]
    singles = [s.copy() for s in slabs]
    e_single = [calc.get_potential_energy(s) for s in singles]
    with caplog.at_level(logging.DEBUG, logger='mcpy.calculators'):
        e_batch = calc.get_potential_energies(slabs)
    np.testing.assert_allclose(e_batch, e_single, rtol=0, atol=1e-6)
    for s, single, f0 in zip(slabs, singles, fixed0):
        np.testing.assert_allclose(s.positions, single.positions, rtol=0, atol=1e-5)
        assert np.array_equal(s.positions[:4], f0)
    assert any(r.getMessage().startswith(f'{optimizer} relaxation')
               for r in caplog.records)


@gpu
def test_step_cap_warning_names_the_optimizer(alchemi, kw, caplog):
    calc = _relax_calc(alchemi, kw, 'fire2', steps=1, fmax=1e-6)
    with caplog.at_level(logging.WARNING, logger='mcpy.calculators'):
        calc.get_potential_energy(_fixed_ag_slab())
    assert any('1-step fire2 cap' in r.getMessage() for r in caplog.records)


@gpu
@pytest.mark.parametrize('optimizer', OPTIMIZERS)
def test_converged_structure_takes_no_step(alchemi, kw, optimizer):
    # calculators-steps-before-convergence-check: ASE checks fmax first.
    calc = _relax_calc(alchemi, kw, optimizer, steps=300)
    slab = _fixed_ag_slab()
    e_relaxed = calc.get_potential_energy(slab)
    relaxed = slab.positions.copy()
    assert calc.get_potential_energy(slab) == pytest.approx(e_relaxed, abs=1e-8)
    assert calc.last_relax_steps == 0
    assert np.array_equal(slab.positions, relaxed)


@gpu
def test_graph_energies_sum_in_float64(alchemi, torch):
    # calculators-f32-energy-reduction: a float32 per-graph sum rounds to meV.
    atomic = torch.full((4000,), -2.7123457, dtype=torch.float32)
    batch = type('B', (), dict(num_graphs=2, batch_idx=torch.arange(4000) // 3000))
    e = alchemi._graph_energies({'atomic_energies': atomic}, batch)
    exact = atomic.double()
    assert e[0] == exact[:3000].sum().item() and e[1] == exact[3000:].sum().item()
    assert abs(atomic[:3000].sum().item() - e[0]) > 1e-4  # what float32 gives


@gpu
def test_energy_is_float64_sum_of_atomic_energies(alchemi, kw, torch):
    calc32 = alchemi.AlchemiCalculator(str(CHECKPOINT), **dict(kw, dtype=torch.float32))
    big = fcc111('Ag', size=(8, 8, 6), a=4.165, vacuum=6.0)
    big.rattle(0.05, seed=5)
    batch = calc32._batch(big)
    alchemi._build_nl(batch, calc32._nl_hook())
    out = calc32._forward(batch)
    ref = out['atomic_energies'].detach().double().sum().item()
    # Up to GPU run-to-run noise; the float32 sum was off by 2.6e-4 eV here.
    assert calc32.get_potential_energy(big) == pytest.approx(ref, abs=2e-5)


@gpu
def test_atoms_info_does_not_reach_the_batch(energy_calc64):
    # calculators-from-atoms-info-leak
    a, b = _fixed_ag_slab(1), _fixed_ag_slab(2)
    ref = [energy_calc64.get_potential_energy(x) for x in (a, b)]
    a.info.update(energy=-1.0, stress=np.zeros(6), tags=['x', 'y'])
    np.testing.assert_allclose(energy_calc64.get_potential_energies([a, b]), ref,
                               rtol=0, atol=1e-8)


@gpu
@pytest.mark.parametrize('path', ['/no/such/dir/model', 'typo.model'])
def test_missing_checkpoint_path_is_file_not_found(alchemi, kw, path):
    # calculators-missing-path-error
    with pytest.raises(FileNotFoundError, match='not found'):
        alchemi.AlchemiCalculator(path, head='omat_pbe', **kw)


@pytest.fixture(scope='module')
def md_calc(alchemi, kw):
    return alchemi.AlchemiFCalculator(str(CHECKPOINT), **kw)


@gpu
def test_md_noise_is_independent_of_initial_velocities(md_calc, tmp_path):
    # calculators-md-seed-reuse: with friction*dt >> 1 the O-step replaces v
    # by fresh noise; noise drawn from the velocity seed copies v0 back.
    atoms = Octahedron('Cu', 5, 1)
    atoms.center(vacuum=8.0)
    atoms.pbc = True
    path = tmp_path / 'md.xyz'
    md_calc.run_md(atoms, temperature=2000.0, friction=10.0, dt=0.5, steps=1,
                   seed=7, trajectory_file=str(path))
    v0, v1 = (f.get_velocities().ravel() for f in read(path, index=':'))
    assert abs(np.corrcoef(v0, v1)[0, 1]) < 0.1


@gpu
def test_md_frozen_rows_stay_put_during_force_evaluation(
        alchemi, md_calc, monkeypatch, tmp_path):
    # calculators-md-frozen-initial-velocity: on nvalchemi 0.2.0 the frozen
    # rows kept their initial velocities and, with the Langevin noise, moved
    # ~0.01 A before every force evaluation.
    from nvalchemi.dynamics.base import DynamicsStage
    seen = []

    class Probe:
        frequency = 1
        stage = DynamicsStage.BEFORE_COMPUTE

        def __call__(self, ctx, stage):
            seen.append(ctx.batch.positions[:4].detach().cpu().numpy().copy())

    class ProbedLangevin(alchemi.NVTLangevin):
        def __init__(self, *args, hooks, **kwargs):
            super().__init__(*args, hooks=hooks + [Probe()], **kwargs)

    monkeypatch.setattr(alchemi, 'NVTLangevin', ProbedLangevin)
    slab = _fixed_ag_slab()
    start = slab.positions[:4].copy()
    path = tmp_path / 'md.xyz'
    md_calc.run_md(slab, temperature=1000.0, steps=3, seed=3,
                   trajectory_file=str(path))
    v0 = read(path, index=0).get_velocities()
    assert np.array_equal(v0[:4], np.zeros((4, 3)))
    assert np.abs(v0[4:]).max() > 0
    # The bootstrap neighbour-list build also fires BEFORE_COMPUTE.
    assert len(seen) >= 3
    for pos in seen:
        np.testing.assert_allclose(pos, start, rtol=0, atol=1e-6)


@gpu
def test_cueq_load_keeps_the_default_dtype(alchemi, torch, device):
    # MACE's cuEq conversion set the process-wide default dtype to float64
    # for good, and a later float32 load of the same model then failed.
    if device != 'cuda':
        pytest.skip('cuEq needs CUDA')
    before = torch.get_default_dtype()  # other tests' MACECalculator sets it
    other = torch.float32 if before == torch.float64 else torch.float64
    alchemi.AlchemiCalculator(str(CHECKPOINT), device=device, dtype=other,
                              enable_cueq=True, compile_model=False)
    assert torch.get_default_dtype() == before


@gpu
@pytest.mark.parametrize('compile_model', [False, True])
def test_local_checkpoint_wrapper_is_inference_only(alchemi, kw, compile_model):
    # A local .model load builds MACEWrapper directly; in train mode every
    # force pass would build a retained second-order graph.
    model = alchemi._load_model(str(CHECKPOINT), kw['device'], kw['dtype'],
                                enable_cueq=False, compile_model=compile_model)
    assert not model.training
    assert not any(m.training for m in model.modules())
    assert not any(p.requires_grad for p in model.parameters())
