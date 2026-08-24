"""Golden end-to-end test: O GCMC on Ag(111) with the production GPU stack.

Opt-in, not CI: needs a GPU, nvalchemi, and a local MACE checkpoint. Run with

    MCPY_GPU_TESTS=1 /path/to/envs/alchemi/bin/python -m pytest \
        tests/test_ag111_gcmc_golden.py -q

The gate is an environment variable rather than an import probe on purpose:
importing torch in an env where it is misconfigured raises SIGBUS, which no
``try/except`` can catch, so a probe would take the whole collection down.

What this covers that the torch-free suite cannot: the Alchemi FIRE relaxation
with ``FreezeAtomsHook``, the real O/Ag(111) energetics, and the fact that the
whole GCMC loop (CustomCell free volume with periodic images, ``min_insert``
overlap rejection, de Broglie acceptance, arrays+constraints rollback,
trajectory writing) reproduces exactly when seeded.

Parameters come from ``examples/03_gcmc_surface_mace.py``, the calibrated set for this
checkpoint: a=4.165, the O-Ag exclusion radius 2.068 used both as the
radius and as the region depth below the top layer, height 7, min_insert 0.5,
and the -0.176 bulk Ag correction.
"""
import os
import pathlib

import numpy as np
import pytest
from ase.build import fcc111
from ase.constraints import FixAtoms
from ase.geometry import find_mic
from ase.io import read

from mcpy.cell import CustomCell
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, InsertionMove
from mcpy.moves.move_selector import MoveSelector

AG_LATTICE = 4.165
R_O_AG = 2.068          # O-Ag exclusion radius, and the region depth below z_top
CELL_HEIGHT = 7.0
MIN_INSERT = 0.5
SEED = 777
STEPS = 60

CHECKPOINT = pathlib.Path(os.environ.get(
    'MCPY_TEST_CHECKPOINT',
    pathlib.Path.home() / '.cache/mace/macesmalldensityagnesistressmodel'))

# Derived from this checkpoint by ``derive_mu_bulk``/``derive_mu_gas``; stable
# across runs, so they are pinned rather than re-derived on every invocation
# (that would add eight relaxations per run). A different checkpoint changes
# these, and the goldens below must then be regenerated together with them.
MU_AG = -2.9929         # derive_mu_bulk('Ag', a=4.165) - 0.176
MU_O = -4.9093          # derive_mu_gas('O2') = E(O2)/2
DELTA_MU_O = -0.5       # balance point: both insertion and deletion accept

# Golden values: seed 777, 60 steps, 4x4x3 slab, bottom layer fixed.
# Energies are float32 GPU sums, so they repeat to ~1e-4 eV rather than
# bitwise; every discrete decision repeats exactly. Regenerate only on a
# deliberate physics change, never by pasting whatever a new run produced.
GOLDEN_N = [48, 53, 53, 51, 52, 52, 53]
GOLDEN_ACCEPTED = [5, 10]
GOLDEN_ENERGY = -151.923111

pytestmark = pytest.mark.skipif(
    not os.environ.get('MCPY_GPU_TESTS') or not CHECKPOINT.exists(),
    reason='set MCPY_GPU_TESTS=1 and provide a MACE checkpoint (GPU required)')


def _build(seed_cell):
    atoms = fcc111('Ag', a=AG_LATTICE, size=(4, 4, 3), periodic=True, vacuum=10.0)
    # Bottom layer only: the two upper layers stay free to relax and to open up
    # for subsurface O.
    atoms.set_constraint(FixAtoms(indices=[a.index for a in atoms if a.tag == 3]))
    z_top = float(atoms.positions[atoms.get_tags() == 1, 2].max())
    cell = CustomCell(atoms, custom_height=CELL_HEIGHT, bottom_z=z_top - R_O_AG,
                      species_radii={'Ag': R_O_AG, 'O': 0.0}, seed=seed_cell)
    return atoms, cell, z_top


def _run(tmp_path, steps=STEPS, write_interval=10):
    from mcpy.calculators import AlchemiFCalculator

    tmp_path.mkdir(parents=True, exist_ok=True)
    seed_del, seed_ins, seed_cell, seed_sel, seed_ens = (
        int(s) for s in np.random.SeedSequence(SEED).generate_state(5, dtype=np.uint32)
    )
    atoms, cell, _ = _build(seed_cell)

    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(cell, species=['O'], seed=seed_del),
         InsertionMove(cell, species=['O'], min_insert=MIN_INSERT, seed=seed_ins)],
        seed=seed_sel,
    )
    calculator = AlchemiFCalculator(
        checkpoint=str(CHECKPOINT), steps=300, fmax=0.05,
        device='cuda', compile_model=False, optimizer='fire2',
    )
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[cell],
        calculator=calculator,
        mu={'Ag': MU_AG, 'O': MU_O + DELTA_MU_O},
        units_type='metal',
        species=['O'],
        temperature=500.0,
        move_selector=move_selector,
        random_seed=seed_ens,
        traj_file=str(tmp_path / 'traj.xyz'),
        outfile=str(tmp_path / 'out.out'),
        trajectory_write_interval=write_interval,
        outfile_write_interval=write_interval,
    )
    gcmc.run(steps)
    return gcmc


@pytest.fixture(scope='module')
def run(tmp_path_factory):
    """One run shared by every assertion below: it costs minutes, not seconds."""
    return _run(tmp_path_factory.mktemp('gcmc'))


def test_golden_trajectory(run):
    """Every accept/reject decision is pinned by the seeds."""
    counts = [len(f) for f in read(run._traj_file, ':')]
    assert counts[:len(GOLDEN_N)] == GOLDEN_N
    assert run.move_selector.move_acceptance_total == GOLDEN_ACCEPTED
    assert run.E_old == pytest.approx(GOLDEN_ENERGY, abs=1e-3)


def test_both_move_types_accept(run):
    """A fixture where one branch never fires stops testing that branch."""
    accepted_del, accepted_ins = run.move_selector.move_acceptance_total
    assert accepted_del > 0, 'no deletion accepted: mu drifted off the balance point'
    assert accepted_ins > 0, 'no insertion accepted: mu drifted off the balance point'


def test_fixed_layer_never_moves(run):
    """FixAtoms must hold across accepted moves, rejected moves and relaxation.

    Compared under the minimum-image convention, not on raw coordinates:
    accepted moves wrap the cell, so an atom crossing x=0 shows a box-sized
    coordinate difference while never having moved.
    """
    atoms = run.atoms
    fixed = atoms.constraints[0].index
    pristine, _, _ = _build(0)
    drift = find_mic(atoms.positions[fixed] - pristine.positions[fixed],
                     atoms.cell, pbc=atoms.pbc)[0]
    assert np.linalg.norm(drift, axis=1).max() < 1e-9


def test_only_oxygen_is_exchanged(run):
    """Ag is not in the move species list, so its count must never change."""
    symbols = np.asarray(run.atoms.get_chemical_symbols())
    assert (symbols == 'Ag').sum() == 48
    assert len(run.atoms) == 48 + (symbols == 'O').sum()


def test_oxygen_stays_deletable(run):
    """An O outside the exchangeable region can never be removed again."""
    atoms = run.atoms
    o_idx = np.where(np.asarray(atoms.get_chemical_symbols()) == 'O')[0]
    assert len(o_idx) > 0
    assert all(run.cells[0].is_point_exchangeable(atoms.positions[i]) for i in o_idx)


def test_oxygen_chemisorbs(run):
    """Real energetics: O binds at a sane distance instead of collapsing into
    the slab or recombining into O2."""
    atoms = run.atoms
    symbols = np.asarray(atoms.get_chemical_symbols())
    o_idx = np.where(symbols == 'O')[0]
    d = atoms.get_all_distances(mic=True)
    assert d[np.ix_(o_idx, np.where(symbols == 'Ag')[0])].min() > 1.8
    if len(o_idx) > 1:
        assert min(d[i][j] for i in o_idx for j in o_idx if i < j) > 2.0


def test_free_volume_is_non_degenerate(run):
    """A collapsed free volume silently breaks the deletion prefactor
    N*Lambda^3/V (inf or nan, see BaseCell._clamp_free_volume)."""
    cell = run.cells[0]
    assert 0.0 < cell.get_volume() < cell.cell_volume


# ------------------------------------------------- batched replica exchange

# The single-GPU flagship: every replica's trial energy in one batched forward
# pass, and the general swap rule that works on both ladders. A mu ladder
# (shared temperature) is used because that is the mode the swap rule's
# cross-terms exist for -- the shortened (beta_j - beta_i)(Phi_j - Phi_i) form
# is identically zero there and would accept every swap.
#
# Spacing is 0.05 eV, deliberately tight: uniform 0.2 eV steps kill the ladder
# (docs/replica_exchange_ladder_spacing.rst) and the test would then only prove
# that nothing swaps.
RE_DELTA_MU_LADDER = [-0.45, -0.50, -0.55]
RE_STEPS = 30
RE_EXCHANGE_INTERVAL = 5

# Golden values: seed 777, ladder above, 30 steps.
RE_GOLDEN_N = [56, 53, 50]                  # final atom count per replica
RE_GOLDEN_O = [8, 5, 2]                     # O count per replica, monotonic in mu
RE_GOLDEN_ENERGY = [-167.4708, -151.8001, -136.2122]


def _replica_factory(tmp_path, calculator):
    """``gcmc_factory(mu=..., rank=...)`` for BatchedReplicaExchange.

    Every replica gets its own atoms, cells, move_selector and seed stream:
    sharing any of them silently mixes the replicas' volumes and counters, and
    the class rejects a factory that does (see its module docstring).
    """
    def factory(mu, rank):
        seeds = np.random.SeedSequence(SEED + rank).generate_state(5, dtype=np.uint32)
        seed_del, seed_ins, seed_cell, seed_sel, seed_ens = (int(s) for s in seeds)
        atoms, cell, _ = _build(seed_cell)
        move_selector = MoveSelector(
            [1, 1],
            [DeletionMove(cell, species=['O'], seed=seed_del),
             InsertionMove(cell, species=['O'], min_insert=MIN_INSERT, seed=seed_ins)],
            seed=seed_sel,
        )
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[cell],
            calculator=calculator,
            mu=mu,
            units_type='metal',
            species=['O'],
            temperature=500.0,
            move_selector=move_selector,
            random_seed=seed_ens,
            traj_file=str(tmp_path / f'replica_{rank}.xyz'),
            outfile=str(tmp_path / f'replica_{rank}.out'),
            trajectory_write_interval=10,
            outfile_write_interval=10,
        )
    return factory


@pytest.fixture(scope='module')
def batched_re(tmp_path_factory):
    """One batched replica-exchange run, shared by the assertions below."""
    from mcpy.calculators import AlchemiFCalculator
    from mcpy.ensembles.batched_replica_exchange import BatchedReplicaExchange

    tmp_path = tmp_path_factory.mktemp('batched_re')
    calculator = AlchemiFCalculator(
        checkpoint=str(CHECKPOINT), steps=300, fmax=0.05,
        device='cuda', compile_model=False, optimizer='fire2',
    )
    re = BatchedReplicaExchange(
        gcmc_factory=_replica_factory(tmp_path, calculator),
        calculator=calculator,
        mus=[{'Ag': MU_AG, 'O': MU_O + d} for d in RE_DELTA_MU_LADDER],
        gcmc_steps=RE_STEPS,
        exchange_interval=RE_EXCHANGE_INTERVAL,
        outfile=str(tmp_path / 're.log'),
        write_out_interval=10,
        seed=SEED,
        global_minimum_file=str(tmp_path / 'global_minimum.xyz'),
    )
    re.run()
    return re, tmp_path


def test_batched_re_golden(batched_re):
    """Per-replica outcome is pinned by the seeds.

    A swap moves a whole configuration between slots, so a single flipped
    borderline acceptance changes which config a slot holds -- the atom counts
    would break first, well before the energy tolerance mattered.
    """
    re, _ = batched_re
    assert [len(r.atoms) for r in re.replicas] == RE_GOLDEN_N
    for r, expected in zip(re.replicas, RE_GOLDEN_ENERGY):
        assert r.E_old == pytest.approx(expected, abs=1e-2)


def test_batched_re_actually_swaps(batched_re):
    """A ladder that never swaps is not replica exchange, it is N lone runs."""
    re, _ = batched_re
    assert sum(re.exchange_attempts) > 0, 'no swap attempted: check exchange_interval'
    assert sum(re.exchange_successes) > 0, (
        'no swap accepted: the mu ladder spacing is too wide '
        '(docs/replica_exchange_ladder_spacing.rst)')


def test_batched_re_replicas_stay_independent(batched_re):
    """Shared atoms, cells or move_selector would silently mix the replicas."""
    re, _ = batched_re
    assert len({id(r.atoms) for r in re.replicas}) == len(re.replicas)
    assert len({id(r.move_selector) for r in re.replicas}) == len(re.replicas)
    cell_ids = [id(c) for r in re.replicas for c in r.cells]
    assert len(set(cell_ids)) == len(cell_ids)


def test_batched_re_each_replica_keeps_its_mu(batched_re):
    """mu is pinned to the slot: a swap moves the configuration, not the mu."""
    re, _ = batched_re
    assert [r._mu['O'] for r in re.replicas] == [MU_O + d for d in RE_DELTA_MU_LADDER]


def test_batched_re_conserves_silver(batched_re):
    """Only O is exchanged, in every replica, swaps included."""
    re, _ = batched_re
    for r in re.replicas:
        symbols = np.asarray(r.atoms.get_chemical_symbols())
        assert (symbols == 'Ag').sum() == 48


def test_batched_re_writes_global_minimum(batched_re):
    """The lowest grand potential across all replicas is written once."""
    re, tmp_path = batched_re
    best = tmp_path / 'global_minimum.xyz'
    assert best.exists()
    frames = read(str(best), ':')
    assert len(frames) == 1
    assert min(r._best_score for r in re.replicas if r._best_atoms is not None) < float('inf')


def test_batched_re_coverage_follows_mu(batched_re):
    """Thermodynamic sanity: a higher mu_O must hold more O.

    This is the assertion a broken ladder fails while everything else still
    looks plausible -- the all-accept degeneracy averages the rungs together
    and flattens exactly this gradient (BatchedReplicaExchange warns about it,
    but only as a post-mortem).
    """
    re, _ = batched_re
    o_counts = [sum(1 for s in r.atoms.get_chemical_symbols() if s == 'O')
                for r in re.replicas]
    assert o_counts == RE_GOLDEN_O
    assert o_counts == sorted(o_counts, reverse=True), (
        f'coverage not monotonic in mu_O: {o_counts} for '
        f'delta_mu {RE_DELTA_MU_LADDER}')
