"""Energy-only GCMC equivalence between the ASE/MACE and Alchemi calculators.

This is an *integration* test: unlike the rest of ``tests/`` it needs ``torch``,
``mace-torch``, ``nvalchemi`` and (in practice) a GPU, plus the ``medium-mpa-0``
MACE-MP checkpoint. It is skipped automatically when any of those are missing, so
the lightweight CI matrix is unaffected. Run it in the GPU env with::

    pytest tests/test_calculator_gcmc_equivalence.py -v -s

Why a "shadow" comparison instead of two independent runs
---------------------------------------------------------
Running GCMC twice with the same seed -- once per calculator -- and asserting an
identical trajectory is *not* a sound test. MACE's CUDA reductions are not
bit-reproducible run-to-run, so a single borderline Metropolis acceptance flipped
by that jitter makes two trajectories diverge for good (this is true even for the
same calculator run twice). The test would be flaky and uninformative.

Instead the test drives ONE real GCMC run with the ASE/MACE calculator and, on
every energy evaluation, shadow-evaluates the Alchemi calculator on a copy of the
same structure. We then assert the two agree on every one of the N steps the run
visits. If they do, GCMC driven by either calculator would make the same
accept/reject decisions -- "the same result in the same number of steps" --
without coupling the assertion to a single fragile decision boundary.

With the same MACE weights and float64 on both sides the two paths agree to
machine precision (observed max|dE| ~3e-14 eV); the tolerance is kept well above
that for hardware slack. Note this covers the energy-only path only: under FIRE
relaxation the ASE and nvalchemi optimisers take different paths and relaxed
energies do not match per-structure (optimiser-path, not model, error).
"""
import os

import numpy as np
import pytest
from ase.cluster import Octahedron

torch = pytest.importorskip('torch')
pytest.importorskip('mace')
pytest.importorskip('nvalchemi')

from mace.calculators import mace_mp  # noqa: E402

from mcpy.calculators import AlchemiCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402


CHECKPOINT = os.environ.get('MCPY_TEST_CHECKPOINT', 'medium-mpa-0')
DEVICE = os.environ.get('MCPY_TEST_DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')
STEPS = int(os.environ.get('MCPY_TEST_STEPS', '12'))
SEED = int(os.environ.get('MCPY_TEST_SEED', '12345'))
# Energy-only: same weights + float64 on both sides -> bit-identical in
# practice (observed max|dE| ~3e-14 eV). Kept well above that for hardware slack.
TOL_ENERGY = float(os.environ.get('MCPY_TEST_TOL_ENERGY', '1e-3'))


class _ASEEnergyOnly:
    """Energy-only mcpy calculator backed by an ASE calculator (no relaxation)."""

    def __init__(self, ase_calc):
        self.calc = ase_calc

    def get_potential_energy(self, atoms):
        atoms.calc = self.calc
        return atoms.get_potential_energy()


class _PairedCalculator:
    """Drives the GCMC with ``driver``; shadow-evaluates ``shadow`` on a copy of
    every structure and records the ``(driver, shadow)`` energy pairs.

    The shadow is evaluated first, on ``atoms.copy()``, so a relaxing driver
    (which mutates ``atoms`` in place) and the shadow both start from the same
    pre-move geometry. Only the driver's energy is returned, so the GCMC follows
    a single, well-defined trajectory.
    """

    def __init__(self, driver, shadow):
        self.driver = driver
        self.shadow = shadow
        self.pairs = []

    def get_potential_energy(self, atoms):
        e_shadow = self.shadow.get_potential_energy(atoms.copy())
        e_driver = self.driver.get_potential_energy(atoms)
        self.pairs.append((float(e_driver), float(e_shadow)))
        return e_driver


def _build_ensemble(calculator, tmp_path):
    """Small Ag octahedron + O insertion/deletion GCMC, deterministically seeded."""
    ss = np.random.SeedSequence(SEED)
    seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))

    atoms = Octahedron('Ag', 4, 1)
    scell = SphericalCell(atoms, vacuum=3.0, species_radii={'Ag': 2.947, 'O': 0.0},
                          mc_sample_points=20_000, seed=SEED)
    species = ['O']
    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(scell, species=species, seed=seed_del),
         InsertionMove(scell, species=species, min_insert=0.5, seed=seed_ins)],
    )
    mus = {'Ag': -2.99, 'O': -4.91 - 0.5}
    return GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[scell],
        calculator=calculator,
        mu=mus,
        units_type='metal',
        species=species,
        temperature=500.0,
        move_selector=move_selector,
        random_seed=SEED,
        outfile=str(tmp_path / 'gcmc.out'),
        traj_file=str(tmp_path / 'gcmc.xyz'),
    )


def test_gcmc_energy_only_ase_vs_alchemi(tmp_path):
    """Energy-only: single-point ASE/MACE vs Alchemi agree along a GCMC run."""
    ase_calc = _ASEEnergyOnly(
        mace_mp(model=CHECKPOINT, device=DEVICE, default_dtype='float64')
    )
    alchemi_calc = AlchemiCalculator(
        checkpoint=CHECKPOINT, device=DEVICE, dtype=torch.float64,
        enable_cueq=False, compile_model=False,
    )
    paired = _PairedCalculator(driver=ase_calc, shadow=alchemi_calc)
    _build_ensemble(paired, tmp_path).run(STEPS)

    pairs = paired.pairs
    assert pairs, 'no energy evaluations were recorded'
    diffs = np.array([abs(a - b) for a, b in pairs])
    worst = int(diffs.argmax())
    print(f'\n[energy-only] {len(pairs)} evals  median|dE|={np.median(diffs):.4g}  '
          f'max|dE|={diffs.max():.4g} eV  '
          f'(worst: driver={pairs[worst][0]:.4f} shadow={pairs[worst][1]:.4f})')
    assert diffs.max() < TOL_ENERGY, (
        f'energy-only: calculators disagree by {diffs.max():.4g} eV at eval {worst} '
        f'(driver={pairs[worst][0]:.6f}, shadow={pairs[worst][1]:.6f}); tol={TOL_ENERGY} eV'
    )
