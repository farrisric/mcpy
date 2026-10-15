"""LAMMPS fix-gcmc parity benchmark for mcpy GCMC.

Three gated stages (see docs/superpowers/specs/2026-07-04-lammps-gcmc-parity-design.md):
  0  ideal-gas calibration: each code vs analytic <N>, extracts LAMMPS's
     LJ-units de Broglie wavelength lambda_L and fixes the mu mapping
         mu_lammps = mu_mcpy + 3 kT ln(lambda_L)
  1  atomic LJ isotherm (insert/delete/translate both sides)
  2  rigid LJ dimer, exchange-only both sides (mcpy MoleculeInsertion/DeletionMove)

Each mu point is the mean and standard error over REPLICAS independently
seeded runs (40% burn-in each). Per-point lines flag |d<N>| > 2 sigma and
|d<PE>| > 3 sigma for inspection, but a stage passes on a chi-square test of
d/sigma over all its points (p > CHI2_P_MIN for both <N> and <PE>): with 13
points, a per-point 2-sigma gate fails about half the time by chance alone.
Stage 0 uses block averages.

Both codes get the same trial budget and move mix per run: mcpy draws its
moves with equal weight, and LAMMPS runs ``fix gcmc`` with one trial per
move type per step (``1 1 0`` for exchange-only stages, ``1 2 1`` with a
third as many steps for stage 1). Dense states relax slowly under GCMC, so
unequal budgets show up as a spurious gap between the codes.

Usage:
    python benchmark/lammps_gcmc_parity.py --stage all
"""
import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones
from scipy.stats import chi2 as chi2_dist

from _parity_common import LJCalc as _LJCalc
from _parity_common import ZeroCalc, block_stats, compare, run_lammps
from mcpy.cell import Cell
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.moves import (DeletionMove, DisplacementMove, InsertionMove,
                        MoleculeDeletionMove, MoleculeInsertionMove,
                        MoveSelector)

L = 9.0
V = L ** 3
T = 2.0
BETA = 1.0 / T
RC = 3.0
REPLICAS = 4  # independent seeds per mu point
CHI2_P_MIN = 0.01
LMP_DEFAULT = os.path.expanduser('~/miniconda3/envs/mcpy/bin/lmp')


def lambda_lj(mass):
    """LAMMPS fix gcmc sets the de Broglie wavelength to 1 in lj units
    (mass-independent), the same convention as mcpy's LJ unit system.
    Confirmed empirically by the Stage 0 extraction (lambda_L ~ 1.003)."""
    return 1.0


def mu_map(mu_mcpy, lam):
    """mcpy (lambda=1) chemical potential -> LAMMPS chemical potential."""
    return mu_mcpy + 3.0 * T * math.log(lam)


# --------------------------------------------------------------------------
# shared infra
# --------------------------------------------------------------------------

def write_csv(outdir, stage, results):
    """One row per (mu, observable): means, stderrs, diff, verdict."""
    path = os.path.join(outdir, f'stage{stage}_results.csv')
    with open(path, 'w') as f:
        f.write('mu,observable,mcpy_mean,mcpy_err,lammps_mean,lammps_err,'
                'diff,sigma_combined,pass\n')
        for mu, rown, rowe, ok in results:
            for label, row in (('N', rown), ('PE', rowe)):
                f.write(f'{mu},{label},' + ','.join(f'{v:.6g}' for v in row)
                        + f',{ok}\n')
    return path


def read_csv(outdir, stage):
    """Inverse of ``write_csv`` for the <N> rows: ``[(mu, row), ...]``."""
    path = os.path.join(outdir, f'stage{stage}_results.csv')
    if not os.path.exists(path):
        return []
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=None,
                         encoding=None)
    data = data[data['observable'] == 'N']
    return [(r['mu'], (r['mcpy_mean'], r['mcpy_err'], r['lammps_mean'],
                       r['lammps_err'])) for r in data]


def plot_isotherms(outdir):
    """Isotherms on top, mcpy - LAMMPS below with the 2-sigma band, one
    column per stage that has a results CSV in ``outdir``."""
    panels = [(results, title, ylabel) for results, title, ylabel in (
        (read_csv(outdir, 1), 'LJ atoms', r'$\langle N \rangle$'),
        (read_csv(outdir, 2), 'rigid LJ dimer',
         r'$\langle N_\mathrm{mol} \rangle$')) if results]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(panels), figsize=(4.5 * len(panels), 5.4),
                             sharex='col', height_ratios=[2, 1],
                             constrained_layout=True, squeeze=False)
    for col, (results, title, ylabel) in enumerate(panels):
        if len(panels) > 1:
            title = f'({"ab"[col]}) {title}'
        top, bottom = axes[0, col], axes[1, col]
        mus = np.array([r[0] for r in results])
        m, sm, ml, sl = (np.array([r[1][i] for r in results]) for i in range(4))
        top.errorbar(mus, ml, yerr=sl, fmt='o-', color='navy', lw=3, ms=8,
                     capsize=3, label='LAMMPS fix gcmc')
        top.errorbar(mus, m, yerr=sm, fmt='o-', color='crimson', lw=1.5, ms=4,
                     capsize=3, label='mcpy')
        top.set_ylabel(ylabel)
        top.set_title(title + r' ($T^*=2$, $L=9\sigma$)')
        top.legend()
        scomb = np.sqrt(sm ** 2 + sl ** 2)
        bottom.fill_between(mus, -2 * scomb, 2 * scomb, color='0.85',
                            label=r'$\pm 2\sigma$')
        bottom.axhline(0.0, color='0.4', lw=0.8)
        bottom.errorbar(mus, m - ml, yerr=scomb, fmt='o', color='crimson',
                        ms=4, capsize=3)
        bottom.set_xlabel(r'$\mu$ ($\epsilon$)')
        bottom.set_ylabel(r'mcpy $-$ LAMMPS')
        bottom.legend(loc='upper left')
    path = os.path.join(outdir, 'lammps_gcmc_parity_isotherms.png')
    fig.savefig(path, dpi=300)
    print(f'  isotherm plot: {path}')
    return path


# --------------------------------------------------------------------------
# mcpy side
# --------------------------------------------------------------------------

def LJCalc():
    """Truncated-and-shifted LJ (sigma=eps=1) for this benchmark's box.
    Verified against ASE LennardJones by ``check_fastlj_matches_ase``."""
    return _LJCalc(L, RC, shift=True)


def check_fastlj_matches_ase(ntrials=5, natoms=40, seed=99):
    rng = np.random.default_rng(seed)
    fast = LJCalc()
    for _ in range(ntrials):
        a = Atoms(f'H{natoms}',
                  positions=rng.uniform(0, L, size=(natoms, 3)),
                  cell=[L, L, L], pbc=True)
        a.calc = LennardJones(sigma=1.0, epsilon=1.0, rc=RC, smooth=False)
        e_ase = a.get_potential_energy()
        e_fast = fast.get_potential_energy(a)
        if abs(e_fast - e_ase) > 1e-9 * max(1.0, abs(e_ase)):
            raise RuntimeError(
                f'fast LJ deviates from ASE: {e_fast} vs {e_ase}')
    print('  fast LJ matches ASE LennardJones on random configs')


def run_mcpy(calc, moves, probs, mu, species, molecules, steps, seed,
             initial_atoms=None):
    atoms = initial_atoms if initial_atoms is not None else \
        Atoms(cell=[L, L, L], pbc=True)
    ms = MoveSelector(probs, moves, seed=seed + 1, n_moves=1)
    g = GrandCanonicalEnsemble(
        atoms=atoms, cells=[Cell(atoms, seed=seed + 2)], units_type='LJ', calculator=calc,
        mu=mu, species=species, temperature=T, move_selector=ms,
        molecules=molecules, random_seed=seed, traj_file=None, outfile=None)
    n_series = np.empty(steps)
    e_series = np.empty(steps)
    for i in range(steps):
        g.do_gcmc_step()
        n_series[i] = len(g.atoms)
        e_series[i] = g.E_old
    return n_series, e_series


def seeded_atoms(n, seed):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.9, L - 0.9, size=(n, 3))
    return Atoms(f'H{n}', positions=pos, cell=[L, L, L], pbc=True)


# --------------------------------------------------------------------------
# stage 0: ideal-gas calibration
# --------------------------------------------------------------------------

def stage0(outdir, lmp, steps_mcpy):
    print('== Stage 0: ideal-gas calibration ==')
    mu0 = -4.0
    n_exact = V * math.exp(BETA * mu0)  # mcpy: lambda = 1
    print(f'  analytic <N>_mcpy = {n_exact:.3f}')

    cell = Cell(Atoms(cell=[L, L, L], pbc=True), seed=10)
    moves = [InsertionMove(cell, species=['H'], seed=11),
             DeletionMove(cell, species=['H'], seed=12)]
    n_ser, _ = run_mcpy(ZeroCalc(), moves, [1, 1], {'H': mu0}, ['H'], None,
                        steps_mcpy, seed=13)
    m, s = block_stats(n_ser)
    ok_mcpy = abs(m - n_exact) < 2 * s
    print(f'  mcpy   <N> = {m:.3f} ± {s:.3f}  vs analytic {n_exact:.3f}  '
          f'({abs(m - n_exact) / s:.2f} sigma)  {"PASS" if ok_mcpy else "FAIL"}')

    lam_pred = lambda_lj(1.0)
    mu_l = mu_map(mu0, lam_pred)
    deck = f"""units lj
atom_style atomic
boundary p p p
region box block 0 {L} 0 {L} 0 {L}
create_box 1 box
mass 1 1.0
pair_style zero {RC}
pair_coeff * *
neighbor 0.3 bin
fix g all gcmc 1 1 0 1 4242 {T} {mu_l} 0.0
thermo 5
thermo_style custom step atoms pe
run {steps_mcpy}
"""
    th = run_lammps(deck, outdir, 'stage0', lmp)
    ml, sl = block_stats(th['atoms'])
    # extract lambda_L from the LAMMPS run
    lam_extracted = (V * math.exp(BETA * mu_l) / ml) ** (1.0 / 3.0)
    ok_lmp = abs(ml - n_exact) < max(2 * sl, 0.01 * n_exact)
    dev = abs(lam_extracted - lam_pred) / lam_pred
    print(f'  lammps <N> = {ml:.3f} ± {sl:.3f}  (mapped mu={mu_l:.4f}, '
          f'expected {n_exact:.3f})  {"PASS" if ok_lmp else "FAIL"}')
    print(f'  lambda_L extracted {lam_extracted:.5f} vs analytic {lam_pred:.5f} '
          f'({100 * dev:.2f}% dev)  {"PASS" if dev < 0.01 else "FAIL"}')
    ok = ok_mcpy and ok_lmp and dev < 0.01
    print(f'  Stage 0: {"PASS" if ok else "FAIL"}')
    return ok, lam_extracted


# --------------------------------------------------------------------------
# potential parity probe
# --------------------------------------------------------------------------

def probe_potential(outdir, lmp):
    """Two-particle energies, ASE vs LAMMPS, for shift yes/no. Returns the
    matching pair_modify shift setting or None."""
    print('== Potential parity probe ==')
    rs = [0.9, 1.0, 1.5, 2.5, 2.99]
    ase_e = []
    for r in rs:
        a = Atoms('H2', positions=[[5, 5, 5], [5, 5, 5 + r]],
                  cell=[20, 20, 20], pbc=True)
        a.calc = LennardJones(sigma=1.0, epsilon=1.0, rc=RC, smooth=False)
        ase_e.append(a.get_potential_energy())
    for shift in ('yes', 'no'):
        ok = True
        for r, ea in zip(rs, ase_e):
            deck = f"""units lj
atom_style atomic
boundary p p p
region box block 0 20 0 20 0 20
create_box 1 box
mass 1 1.0
pair_style lj/cut {RC}
pair_coeff 1 1 1.0 1.0
pair_modify shift {shift}
create_atoms 1 single 5 5 5
create_atoms 1 single 5 5 {5 + r}
thermo_style custom step atoms pe
run 0
"""
            th = run_lammps(deck, outdir, f'probe_{shift}_{r}', lmp)
            el = th['pe'][-1] * 2  # thermo pe is per-atom by default in lj units
            if abs(el - ea) > 1e-8 * max(1.0, abs(ea)):
                ok = False
                break
        if ok:
            print(f'  ASE LennardJones matches pair_modify shift {shift} '
                  f'(checked r={rs})')
            return shift
    print('  FAIL: no LAMMPS shift convention matches ASE energies')
    print(f'    ASE energies: {dict(zip(rs, ase_e))}')
    return None


# --------------------------------------------------------------------------
# stage 1: atomic LJ isotherm
# --------------------------------------------------------------------------

def _stage1_point(outdir, lmp, lam, shift, steps_mcpy, mu, rep):
    mu_l = mu_map(mu, lam)
    k = 100 * rep
    cell = Cell(Atoms(cell=[L, L, L], pbc=True), seed=20 + k)
    moves = [InsertionMove(cell, species=['H'], seed=21 + k),
             DeletionMove(cell, species=['H'], seed=22 + k, min_atoms=1),
             DisplacementMove(species=['H'], seed=23 + k, max_displacement=0.3)]
    n_ser, e_ser = run_mcpy(
        LJCalc(), moves, [1, 1, 1], {'H': mu}, ['H'], None, steps_mcpy,
        seed=24 + k, initial_atoms=seeded_atoms(20, 25 + k))

    deck = f"""units lj
atom_style atomic
boundary p p p
region box block 0 {L} 0 {L} 0 {L}
create_box 1 box
mass 1 1.0
pair_style lj/cut {RC}
pair_coeff 1 1 1.0 1.0
pair_modify shift {shift}
neighbor 0.3 bin
fix g all gcmc 1 2 1 1 {777 + k} {T} {mu_l} 0.3
thermo 5
thermo_style custom step atoms pe
run {steps_mcpy // 3}
"""
    th = run_lammps(deck, outdir, f'stage1_mu{mu}_r{rep}', lmp)
    # thermo pe in lj units is per-atom (thermo_modify norm yes default):
    # convert to extensive using the atom count series.
    return (_burned_mean(n_ser), _burned_mean(e_ser),
            _burned_mean(th['atoms']), _burned_mean(th['pe'] * th['atoms']))


def stage1(outdir, lmp, lam, shift, steps_mcpy, mus):
    print('== Stage 1: atomic LJ isotherm ==')
    return _run_points(_stage1_point, '<N>',
                       (outdir, lmp, lam, shift, steps_mcpy), mus,
                       'Stage 1')


def _burned_mean(series, burn_frac=0.4):
    x = np.asarray(series, dtype=float)
    return x[int(len(x) * burn_frac):].mean()


def _run_points(point_fn, nlabel, fixed, mus, name, nrep=REPLICAS):
    """Run ``nrep`` independently seeded replicas of every mu point in
    parallel. The error bar is the standard error across replicas: a single
    run's block average underestimates it in the dense fluid, where N
    decorrelates slowly."""
    jobs = [(mu, rep) for mu in mus for rep in range(nrep)]
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count())) as ex:
        futures = [ex.submit(point_fn, *fixed, mu, rep) for mu, rep in jobs]
        runs = np.array([f.result() for f in futures]).reshape(len(mus), nrep, 4)
    means = runs.mean(axis=1)
    errs = runs.std(axis=1, ddof=1) / math.sqrt(nrep)
    results = []
    for mu, m, e in zip(mus, means, errs):
        print(f'  mu={mu}')
        okn, rown = compare(nlabel, m[0], e[0], m[2], e[2], 2)
        oke, rowe = compare('<PE>', m[1], e[1], m[3], e[3], 3)
        results.append((mu, rown, rowe, okn and oke))
    all_ok = True
    for label, col in ((nlabel, 1), ('<PE>', 2)):
        chi2 = sum((r[col][4] / r[col][5]) ** 2 for r in results)
        pval = chi2_dist.sf(chi2, len(results))
        ok = pval > CHI2_P_MIN
        all_ok &= ok
        print(f'  {label} chi2 = {chi2:.1f} over {len(results)} points, '
              f'p = {pval:.3f}  {"PASS" if ok else "FAIL"}')
    print(f'  {name}: {"PASS" if all_ok else "FAIL"}')
    return all_ok, results


# --------------------------------------------------------------------------
# stage 2: rigid dimer, exchange-only
# --------------------------------------------------------------------------

DIMER_MOL = """# rigid LJ dimer (no bonds: intra pair counted by pair_style)

2 atoms

Coords

1 0.0 0.0 0.0
2 0.0 0.0 1.0

Types

1 1
2 1

Masses

1 1.0
2 1.0
"""


def _stage2_point(outdir, lmp, lam, shift, steps_mcpy, mu, rep):
    # lj units: lambda = 1 in both codes regardless of mass (mcpy _set_lj_units,
    # LAMMPS fix gcmc lj convention) -- no mass correction.
    mu_l = mu_map(mu, lam)
    k = 100 * rep
    template = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
    cell = Cell(Atoms(cell=[L, L, L], pbc=True), seed=30 + k)
    moves = [MoleculeInsertionMove(cell, template, 'H2', seed=31 + k),
             MoleculeDeletionMove(cell, template, 'H2', seed=32 + k)]
    n_ser, e_ser = run_mcpy(
        LJCalc(), moves, [1, 1], {'H2': mu}, [], {'H2': template},
        steps_mcpy, seed=33 + k)

    deck = f"""units lj
atom_style molecular
boundary p p p
region box block 0 {L} 0 {L} 0 {L}
create_box 1 box
mass 1 1.0
pair_style lj/cut {RC}
pair_coeff 1 1 1.0 1.0
pair_modify shift {shift}
neighbor 0.3 bin
molecule dimer dimer.mol
fix g all gcmc 1 1 0 0 {888 + k} {T} {mu_l} 0.0 mol dimer
thermo 5
thermo_style custom step atoms pe
run {steps_mcpy}
"""
    th = run_lammps(deck, outdir, f'stage2_mu{mu}_r{rep}', lmp)
    return (_burned_mean(n_ser / 2.0), _burned_mean(e_ser),  # molecules
            _burned_mean(th['atoms'] / 2.0),
            _burned_mean(th['pe'] * th['atoms']))


def stage2(outdir, lmp, lam, shift, steps_mcpy, mus):
    print('== Stage 2: rigid LJ dimer, exchange-only ==')
    with open(os.path.join(outdir, 'dimer.mol'), 'w') as f:
        f.write(DIMER_MOL)
    return _run_points(_stage2_point, '<Nmol>',
                       (outdir, lmp, lam, shift, steps_mcpy), mus,
                       'Stage 2')


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lmp', default=LMP_DEFAULT)
    p.add_argument('--stage', default='all', choices=['all', '0', '1', '2'])
    p.add_argument('--mcpy-steps', type=int, default=200_000,
                   help='trial moves per (stage, mu, replica), same for both codes')
    p.add_argument('--outdir', default='benchmark/lammps_parity')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    lam = lambda_lj(1.0)
    if args.stage in ('all', '0'):
        # stage 0 only checks lambda_L ~ 1; later stages keep the exact
        # lambda = 1 convention rather than the noisy fitted value.
        ok, _ = stage0(args.outdir, args.lmp, args.mcpy_steps)
        if not ok:
            raise SystemExit('Stage 0 FAILED: mu mapping unreliable, aborting.')
        if args.stage == '0':
            return

    shift = probe_potential(args.outdir, args.lmp)
    if shift is None:
        raise SystemExit('Potential parity probe FAILED, aborting.')

    mus1 = [round(-4.0 + 0.2 * i, 1) for i in range(13)]  # -4.0 .. -1.6
    # Dimer grid sits lower: at mu >= -3 the dimer system is a dense liquid
    # (>400 atoms) that exchange-only sampling cannot equilibrate in these run
    # lengths; both codes showed still-climbing N traces there.
    mus2 = [round(-7.0 + 0.25 * i, 2) for i in range(13)]  # -7.0 .. -4.0
    if args.stage in ('all', '1'):
        ok1, s1_results = stage1(args.outdir, args.lmp, lam, shift,
                                 args.mcpy_steps, mus1)
        write_csv(args.outdir, 1, s1_results)
        if not ok1 and args.stage == 'all':
            raise SystemExit('Stage 1 FAILED, aborting before stage 2.')

    if args.stage in ('all', '2'):
        _, s2_results = stage2(args.outdir, args.lmp, lam, shift,
                               args.mcpy_steps, mus2)
        write_csv(args.outdir, 2, s2_results)

    plot_isotherms(args.outdir)


if __name__ == '__main__':
    main()
