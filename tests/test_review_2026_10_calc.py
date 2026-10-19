"""Review 2026-10, calc group: calculator fixes.

The CPU tests never import torch. The Alchemi tests are opt-in
(``MCPY_GPU_TESTS=1``; a torch import probe can SIGBUS) and run on CUDA when
it is there, else on the CPU (``MCPY_TEST_DEVICE`` overrides).
"""
import logging
import os
import pathlib

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.calculators.lj import LennardJones
from ase.cluster import Octahedron
from ase.constraints import FixAtoms
from ase.io import read

from mcpy.calculators import BaseCalculator
from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, InsertionMove, MoveSelector


# ------------------------------------------- gcmc-physics-empty-relax-crash

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


# ------------------------------------------------------- Alchemi (opt-in)

CHECKPOINT = pathlib.Path(os.environ.get(
    'MCPY_TEST_CHECKPOINT',
    pathlib.Path.home() / '.cache/mace/macesmalldensityagnesistressmodel'))

gpu = pytest.mark.skipif(
    not os.environ.get('MCPY_GPU_TESTS') or not CHECKPOINT.exists(),
    reason='set MCPY_GPU_TESTS=1 and provide a MACE checkpoint')


@pytest.fixture(scope='module')
def torch():
    import torch
    return torch


@pytest.fixture(scope='module')
def alchemi():
    from mcpy.calculators import alchemi
    return alchemi


@pytest.fixture(scope='module')
def device(torch):
    return os.environ.get('MCPY_TEST_DEVICE') or (
        'cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope='module')
def kw(torch, device):
    return dict(device=device, dtype=torch.float64, enable_cueq=False,
                compile_model=False)


@pytest.fixture(scope='module')
def energy_calc(alchemi, kw):
    return alchemi.AlchemiCalculator(str(CHECKPOINT), **kw)


@pytest.fixture(scope='module')
def md_calc(alchemi, kw):
    return alchemi.AlchemiFCalculator(str(CHECKPOINT), **kw)


def _ag_slab(seed=1, fixed=True):
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
def test_relax_steps0_is_single_point(alchemi, kw, energy_calc, optimizer):
    # gpu-lbfgs-steps0-crash: nvalchemi main rejects n_steps=0.
    calc = _relax_calc(alchemi, kw, optimizer, steps=0)
    slab = _ag_slab()
    start = slab.positions.copy()
    e = calc.get_potential_energy(slab)
    assert np.array_equal(slab.positions, start)
    assert calc.last_relax_steps == 0
    assert e == pytest.approx(energy_calc.get_potential_energy(slab), abs=1e-8)


@gpu
@pytest.mark.parametrize('optimizer', OPTIMIZERS)
def test_batched_relax_matches_single_and_holds_fixed(alchemi, kw, optimizer, caplog):
    # calculators-lbfgs-untested-cuda-only-tests
    calc = _relax_calc(alchemi, kw, optimizer, steps=300)
    slabs = [_ag_slab(seed) for seed in (1, 2, 3)]
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
        calc.get_potential_energy(_ag_slab())
    assert any('1-step fire2 cap' in r.getMessage() for r in caplog.records)


@gpu
@pytest.mark.parametrize('optimizer', OPTIMIZERS)
def test_converged_structure_takes_no_step(alchemi, kw, optimizer):
    # calculators-steps-before-convergence-check: ASE checks fmax first.
    calc = _relax_calc(alchemi, kw, optimizer, steps=300)
    slab = _ag_slab()
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
def test_atoms_info_does_not_reach_the_batch(energy_calc):
    # calculators-from-atoms-info-leak
    a, b = _ag_slab(1), _ag_slab(2)
    ref = [energy_calc.get_potential_energy(x) for x in (a, b)]
    a.info.update(energy=-1.0, stress=np.zeros(6), tags=['x', 'y'])
    np.testing.assert_allclose(energy_calc.get_potential_energies([a, b]), ref,
                               rtol=0, atol=1e-8)


@gpu
@pytest.mark.parametrize('path', ['/no/such/dir/model', 'typo.model'])
def test_missing_checkpoint_path_is_file_not_found(alchemi, kw, path):
    # calculators-missing-path-error
    with pytest.raises(FileNotFoundError, match='not found'):
        alchemi.AlchemiCalculator(path, head='omat_pbe', **kw)


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
    slab = _ag_slab()
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
