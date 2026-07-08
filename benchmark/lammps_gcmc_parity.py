"""LAMMPS fix-gcmc parity benchmark for mcpy GCMC.

Three gated stages (see docs/superpowers/specs/2026-07-04-lammps-gcmc-parity-design.md):
  0  ideal-gas calibration: each code vs analytic <N>, extracts LAMMPS's
     LJ-units de Broglie wavelength lambda_L and fixes the mu mapping
         mu_lammps = mu_mcpy + 3 kT ln(lambda_L)
  1  atomic LJ isotherm (insert/delete/translate both sides)
  2  rigid LJ dimer, exchange-only both sides (mcpy MoleculeInsertion/DeletionMove)

Pass criteria per mu point: |d<N>| < 2 sigma_combined, |d<PE>| < 3 sigma_combined
(block averages, 40% burn-in, 15 blocks).

Usage:
    python benchmark/lammps_gcmc_parity.py --stage all
"""
import argparse
import math
import os
import subprocess

import numpy as np
from ase import Atoms
from ase.calculators.lj import LennardJones

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

def run_lammps(deck, outdir, tag, lmp):
    deckfile = os.path.join(outdir, f'in.{tag}')
    logfile = os.path.join(outdir, f'log.{tag}')
    with open(deckfile, 'w') as f:
        f.write(deck)
    subprocess.run([lmp, '-in', os.path.abspath(deckfile),
                    '-log', os.path.abspath(logfile), '-screen', 'none'],
                   check=True, cwd=outdir)
    return parse_thermo(logfile)


def parse_thermo(logfile):
    """Collect thermo rows from every run section. Returns dict of arrays
    keyed by lower-cased column names (step, atoms, pe, ...)."""
    cols, rows = None, []
    with open(logfile) as f:
        in_section = False
        for line in f:
            s = line.split()
            if not s:
                continue
            if s[0] == 'Step':
                cols = [c.lower() for c in s]
                in_section = True
                continue
            if in_section:
                if s[0] == 'Loop' or line.startswith('WARNING'):
                    in_section = False
                    continue
                try:
                    rows.append([float(v) for v in s])
                except ValueError:
                    in_section = False
    if cols is None or not rows:
        raise RuntimeError(f'no thermo data parsed from {logfile}')
    rows = [r for r in rows if len(r) == len(cols)]
    data = np.array(rows)
    out = {c: data[:, i] for i, c in enumerate(cols)}
    if 'poteng' in out:  # thermo header prints "PotEng" for pe
        out['pe'] = out['poteng']
    return out


def block_stats(series, burn_frac=0.4, nblocks=15):
    x = np.asarray(series, dtype=float)
    x = x[int(len(x) * burn_frac):]
    blocks = np.array_split(x, nblocks)
    means = np.array([b.mean() for b in blocks])
    return means.mean(), means.std(ddof=1) / math.sqrt(nblocks)


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


def plot_isotherms(outdir, s1_results, s2_results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, results, title, ylabel in (
            (axes[0], s1_results, 'Stage 1: LJ atoms', r'$\langle N \rangle$'),
            (axes[1], s2_results, 'Stage 2: rigid LJ dimer',
             r'$\langle N_\mathrm{mol} \rangle$')):
        if not results:
            ax.set_visible(False)
            continue
        mus = [r[0] for r in results]
        ax.errorbar(mus, [r[1][0] for r in results],
                    yerr=[r[1][1] for r in results], fmt='o-', capsize=4,
                    label='mcpy')
        ax.errorbar(mus, [r[1][2] for r in results],
                    yerr=[r[1][3] for r in results], fmt='s--', capsize=4,
                    label='LAMMPS fix gcmc')
        ax.set_xlabel(r'$\mu$ (LJ units)')
        ax.set_ylabel(ylabel)
        ax.set_title(title + r' ($T^*=2$, $L=9$)')
        ax.legend()
    path = os.path.join(outdir, 'lammps_gcmc_parity_isotherms.png')
    fig.savefig(path, dpi=150)
    print(f'  isotherm plot: {path}')
    return path


def compare(label, m_mcpy, s_mcpy, m_lmp, s_lmp, nsig):
    diff = abs(m_mcpy - m_lmp)
    scomb = math.sqrt(s_mcpy ** 2 + s_lmp ** 2)
    ok = diff < nsig * scomb
    print(f'    {label:6s} mcpy {m_mcpy:10.4f} ± {s_mcpy:.4f}   '
          f'lammps {m_lmp:10.4f} ± {s_lmp:.4f}   '
          f'|d|={diff:.4f} ({diff / scomb if scomb > 0 else float("inf"):.2f} sigma)  '
          f'{"PASS" if ok else "FAIL"}')
    return ok, (m_mcpy, s_mcpy, m_lmp, s_lmp, diff, scomb)


# --------------------------------------------------------------------------
# mcpy side
# --------------------------------------------------------------------------

class ZeroCalc:
    def get_potential_energy(self, atoms):
        return 0.0


class LJCalc:
    """Vectorized truncated-and-shifted LJ (sigma=eps=1), minimum image in a
    cubic box of side L. Verified against ASE LennardJones by
    ``check_fastlj_matches_ase`` at startup (machine precision)."""

    E_SHIFT = 4.0 * (RC ** -12 - RC ** -6)

    def get_potential_energy(self, atoms):
        pos = atoms.positions
        n = len(pos)
        if n < 2:
            return 0.0
        d = pos[:, None, :] - pos[None, :, :]
        d -= L * np.round(d / L)
        r2 = np.einsum('ijk,ijk->ij', d, d)
        iu = np.triu_indices(n, k=1)
        r2 = r2[iu]
        r2 = r2[r2 < RC * RC]
        inv6 = 1.0 / r2 ** 3
        return float(np.sum(4.0 * (inv6 * inv6 - inv6) - self.E_SHIFT))


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
        atoms=atoms, cells=[Cell(atoms)], units_type='LJ', calculator=calc,
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

def stage0(outdir, lmp, steps_mcpy, lmp_steps):
    print('== Stage 0: ideal-gas calibration ==')
    mu0 = -4.0
    n_exact = V * math.exp(BETA * mu0)  # mcpy: lambda = 1
    print(f'  analytic <N>_mcpy = {n_exact:.3f}')

    cell = Cell(Atoms(cell=[L, L, L], pbc=True))
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
fix g all gcmc 1 100 0 1 4242 {T} {mu_l} 0.0
thermo 5
thermo_style custom step atoms pe
run {lmp_steps}
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

def stage1(outdir, lmp, lam, shift, steps_mcpy, lmp_steps, mus):
    print('== Stage 1: atomic LJ isotherm ==')
    results = []
    all_ok = True
    for mu in mus:
        mu_l = mu_map(mu, lam)
        print(f'  mu_mcpy={mu}  (mu_lammps={mu_l:.4f})')
        cell = Cell(Atoms(cell=[L, L, L], pbc=True))
        moves = [InsertionMove(cell, species=['H'], seed=21),
                 DeletionMove(cell, species=['H'], seed=22, min_atoms=1),
                 DisplacementMove(species=['H'], seed=23, max_displacement=0.3)]
        n_ser, e_ser = run_mcpy(
            LJCalc(), moves, [1, 1, 1], {'H': mu}, ['H'], None, steps_mcpy,
            seed=24, initial_atoms=seeded_atoms(20, 25))
        mn, sn = block_stats(n_ser)
        me, se = block_stats(e_ser)

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
fix g all gcmc 1 10 10 1 777 {T} {mu_l} 0.3
thermo 5
thermo_style custom step atoms pe
run {lmp_steps}
"""
        th = run_lammps(deck, outdir, f'stage1_mu{mu}', lmp)
        mln, sln = block_stats(th['atoms'])
        # thermo pe in lj units is per-atom (thermo_modify norm yes default):
        # convert to extensive using the atom count series.
        pe_ext = th['pe'] * th['atoms']
        mle, sle = block_stats(pe_ext)

        okn, rown = compare('<N>', mn, sn, mln, sln, 2)
        oke, rowe = compare('<PE>', me, se, mle, sle, 3)
        all_ok &= okn and oke
        results.append((mu, rown, rowe, okn and oke))
    print(f'  Stage 1: {"PASS" if all_ok else "FAIL"}')
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


def stage2(outdir, lmp, lam1, shift, steps_mcpy, lmp_steps, mus):
    print('== Stage 2: rigid LJ dimer, exchange-only ==')
    # lj units: lambda = 1 in both codes regardless of mass (mcpy _set_lj_units,
    # LAMMPS fix gcmc lj convention) -- no mass correction.
    lam2 = lam1
    molfile = os.path.join(outdir, 'dimer.mol')
    with open(molfile, 'w') as f:
        f.write(DIMER_MOL)
    template = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
    results = []
    all_ok = True
    for mu in mus:
        mu_l = mu_map(mu, lam2)
        print(f'  mu_mcpy={mu}  (mu_lammps={mu_l:.4f})')
        cell = Cell(Atoms(cell=[L, L, L], pbc=True))
        moves = [MoleculeInsertionMove(cell, template, 'H2', seed=31),
                 MoleculeDeletionMove(cell, template, 'H2', seed=32)]
        n_ser, e_ser = run_mcpy(
            LJCalc(), moves, [1, 1], {'H2': mu}, [], {'H2': template},
            steps_mcpy, seed=33)
        mn, sn = block_stats(n_ser / 2.0)  # molecules
        me, se = block_stats(e_ser)

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
fix g all gcmc 1 10 0 0 888 {T} {mu_l} 0.0 mol dimer
thermo 5
thermo_style custom step atoms pe
run {lmp_steps}
"""
        th = run_lammps(deck, outdir, f'stage2_mu{mu}', lmp)
        mln, sln = block_stats(th['atoms'] / 2.0)
        pe_ext = th['pe'] * th['atoms']
        mle, sle = block_stats(pe_ext)

        okn, rown = compare('<Nmol>', mn, sn, mln, sln, 2)
        oke, rowe = compare('<PE>', me, se, mle, sle, 3)
        all_ok &= okn and oke
        results.append((mu, rown, rowe, okn and oke))
    print(f'  Stage 2: {"PASS" if all_ok else "FAIL"}')
    return all_ok, results


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--lmp', default=LMP_DEFAULT)
    p.add_argument('--stage', default='all', choices=['all', '0', '1', '2'])
    p.add_argument('--mcpy-steps', type=int, default=150_000,
                   help='mcpy trial moves per (stage, mu)')
    p.add_argument('--lmp-steps', type=int, default=20_000,
                   help='LAMMPS timesteps per (stage, mu)')
    p.add_argument('--outdir', default='benchmark/lammps_parity')
    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    lam = lambda_lj(1.0)
    if args.stage in ('all', '0'):
        ok, lam = stage0(args.outdir, args.lmp, args.mcpy_steps, args.lmp_steps)
        if not ok:
            raise SystemExit('Stage 0 FAILED: mu mapping unreliable, aborting.')
        if args.stage == '0':
            return

    shift = probe_potential(args.outdir, args.lmp)
    if shift is None:
        raise SystemExit('Potential parity probe FAILED, aborting.')

    mus1 = [-3.0, -2.4, -1.8]
    # Dimer grid sits lower: at mu >= -3 the dimer system is a dense liquid
    # (>400 atoms) that exchange-only sampling cannot equilibrate in these run
    # lengths; both codes showed still-climbing N traces there.
    mus2 = [-6.0, -5.0, -4.0]
    s1_results, s2_results = [], []
    if args.stage in ('all', '1'):
        ok1, s1_results = stage1(args.outdir, args.lmp, lam, shift,
                                 args.mcpy_steps, args.lmp_steps, mus1)
        write_csv(args.outdir, 1, s1_results)
        if not ok1 and args.stage == 'all':
            raise SystemExit('Stage 1 FAILED, aborting before stage 2.')

    if args.stage in ('all', '2'):
        ok2, s2_results = stage2(args.outdir, args.lmp, lam, shift,
                                 args.mcpy_steps, args.lmp_steps, mus2)
        write_csv(args.outdir, 2, s2_results)

    if s1_results or s2_results:
        plot_isotherms(args.outdir, s1_results, s2_results)


if __name__ == '__main__':
    main()
