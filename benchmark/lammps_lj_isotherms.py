"""mcpy vs LAMMPS fix gcmc on two single-species Lennard-Jones isotherms.

Reproduces Figure 1 of the JOSS paper:
  a  purely repulsive LJ fluid (WCA cut at 2^(1/6) sigma), LJ units, T* = 1,
     cubic box L = 7 sigma
  b  argon in metal units (eps = 0.010086 eV, sigma = 3.4 A, cut 3 sigma),
     T = 87.79 K, cubic box L = 27.2 A, across the gas-liquid condensation

Both codes use the unshifted lj/cut (the LAMMPS default) with matching
cutoffs, the same initial state, and the same trial budget and move mix: mcpy
draws insertion, deletion and translation with equal weight; LAMMPS runs
``fix gcmc 1 2 1`` (two exchange attempts and one translation per step) for a
third as many steps. Every mu point is the mean and standard error over
REPLICAS independently seeded runs per code (40% burn-in each); the bottom
panels show mcpy - LAMMPS against the combined 2-sigma band.

Argon liquid: GCMC relaxes the dense liquid slowly in both codes (about one
accepted exchange per thousand attempts), so a run started far from the
plateau ends short of it. Each liquid point therefore starts from a liquid
equilibrated by one long LAMMPS GCMC run at that mu (from an overfilled
8x8x8 lattice), which both codes then sample. Gas points start from an empty
box. The grid skips mu = -8.6 eps, on the coexistence line, where neither
code nucleates reproducibly within the run length.

Usage:
    python benchmark/lammps_lj_isotherms.py            # run + plot
    python benchmark/lammps_lj_isotherms.py --plot     # replot from the CSV
"""
import argparse
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from ase import Atoms
from ase.io import read

from _parity_common import LJCalc, run_lammps
from mcpy.cell import Cell
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector

LMP_DEFAULT = os.path.expanduser('~/miniconda3/envs/mcpy/bin/lmp')
REPLICAS = 4
BURN_FRAC = 0.4
EPS_AR = 0.010086

SYSTEMS = {
    'repulsive': dict(
        units='lj', mass=1.0, box=7.0, eps=1.0, sigma=1.0, rc=2 ** (1 / 6),
        temperature=1.0, displacement=0.3,
        mus=[-2.4, -1.9, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        liquid_from=None),
    'argon': dict(
        units='metal', mass=39.948, box=27.2, eps=EPS_AR, sigma=3.4, rc=10.2,
        temperature=87.79, displacement=1.0,
        mus=[round(m * EPS_AR, 7) for m in
             (-9.0, -8.9, -8.8, -8.7, -8.58, -8.56, -8.54, -8.52, -8.5,
              -8.4, -8.3, -8.2, -8.1, -8.0)],
        liquid_from=-8.59 * EPS_AR),
}


def is_liquid(sys_, mu):
    return sys_['liquid_from'] is not None and mu >= sys_['liquid_from']


def start_file(outdir, name, mu):
    return os.path.join(outdir, f'start_{name}_mu{mu}.data')


def prepare_liquid(name, sys_, mu, outdir, lmp, steps=60_000):
    """Equilibrate the liquid at ``mu`` with a long LAMMPS GCMC run from an
    overfilled 8x8x8 simple-cubic lattice and write it as the shared start."""
    box = sys_['box']
    deck = f"""units {sys_['units']}
atom_style atomic
boundary p p p
region box block 0 {box} 0 {box} 0 {box}
create_box 1 box
lattice sc {box / 8}
create_atoms 1 region box
mass 1 {sys_['mass']}
pair_style lj/cut {sys_['rc']}
pair_coeff 1 1 {sys_['eps']} {sys_['sigma']}
neighbor {0.3 * sys_['sigma']} bin
fix g all gcmc 1 10 10 1 4242 {sys_['temperature']} {mu} {sys_['displacement']}
thermo 100
thermo_style custom step atoms pe
run {steps}
write_data {os.path.basename(start_file(outdir, name, mu))} nocoeff
"""
    run_lammps(deck, outdir, f'prep_{name}_mu{mu}', lmp)


def initial_atoms(sys_, name, mu, outdir):
    """Empty box, or the prepared liquid for liquid-branch points."""
    if not is_liquid(sys_, mu):
        return Atoms(cell=[sys_['box']] * 3, pbc=True)
    atoms = read(start_file(outdir, name, mu), format='lammps-data',
                 atom_style='atomic', units=sys_['units'], Z_of_type={1: 18})
    atoms.wrap()
    return atoms


def run_mcpy(name, sys_, mu, rep, steps, outdir):
    k = 100 * rep
    atoms = initial_atoms(sys_, name, mu, outdir)
    cell = Cell(Atoms(cell=[sys_['box']] * 3, pbc=True), seed=10 + k)
    moves = [InsertionMove(cell, species=['Ar'], seed=11 + k),
             DeletionMove(cell, species=['Ar'], seed=12 + k),
             DisplacementMove(species=['Ar'], seed=13 + k,
                              max_displacement=sys_['displacement'])]
    ms = MoveSelector([1, 1, 1], moves, seed=14 + k, n_moves=1)
    g = GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell],
        units_type={'lj': 'LJ', 'metal': 'metal'}[sys_['units']],
        calculator=LJCalc(sys_['box'], sys_['rc'], sys_['eps'], sys_['sigma']),
        mu={'Ar': mu}, species=['Ar'], temperature=sys_['temperature'],
        move_selector=ms, random_seed=15 + k, traj_file=None, outfile=None)
    n = np.empty(steps)
    for i in range(steps):
        g.do_gcmc_step()
        n[i] = len(g.atoms)
    return n[int(BURN_FRAC * len(n)):].mean()


def lammps_deck(name, sys_, mu, rep, steps, outdir):
    """``steps`` trial moves as ``fix gcmc 1 2 1``: three trials per step."""
    box = sys_['box']
    if is_liquid(sys_, mu):
        setup = f'read_data {os.path.basename(start_file(outdir, name, mu))}\n'
    else:
        setup = (f'region box block 0 {box} 0 {box} 0 {box}\n'
                 f'create_box 1 box\n')
    return f"""units {sys_['units']}
atom_style atomic
boundary p p p
{setup}mass 1 {sys_['mass']}
pair_style lj/cut {sys_['rc']}
pair_coeff 1 1 {sys_['eps']} {sys_['sigma']}
neighbor {0.3 * sys_['sigma']} bin
fix g all gcmc 1 2 1 1 {777 + 100 * rep} {sys_['temperature']} {mu} {sys_['displacement']}
thermo 1
thermo_style custom step atoms pe
run {steps // 3}
"""


def run_lammps_point(name, sys_, mu, rep, steps, outdir, lmp):
    th = run_lammps(lammps_deck(name, sys_, mu, rep, steps, outdir), outdir,
                    f'{name}_mu{mu}_r{rep}', lmp)
    n = th['atoms']
    return n[int(BURN_FRAC * len(n)):].mean()


def job(code, name, mu, rep, args):
    sys_ = SYSTEMS[name]
    if code == 'mcpy':
        return run_mcpy(name, sys_, mu, rep, args.steps, args.outdir)
    return run_lammps_point(name, sys_, mu, rep, args.steps, args.outdir, args.lmp)


def probe_potential(outdir, lmp):
    """Two-body energies, LJCalc vs LAMMPS, for both systems."""
    for name, sys_ in SYSTEMS.items():
        calc = LJCalc(sys_['box'], sys_['rc'], sys_['eps'], sys_['sigma'])
        for frac in (0.95, 1.0, 1.05, 0.98 * sys_['rc'] / sys_['sigma']):
            r = frac * sys_['sigma']
            a = Atoms('Ar2', positions=[[1, 1, 1], [1, 1, 1 + r]],
                      cell=[sys_['box']] * 3, pbc=True)
            box = sys_['box']
            deck = f"""units {sys_['units']}
atom_style atomic
boundary p p p
region box block 0 {box} 0 {box} 0 {box}
create_box 1 box
mass 1 {sys_['mass']}
pair_style lj/cut {sys_['rc']}
pair_coeff 1 1 {sys_['eps']} {sys_['sigma']}
create_atoms 1 single 1 1 1
create_atoms 1 single 1 1 {1 + r}
thermo_style custom step pe
thermo_modify norm no
run 0
"""
            e_lmp = run_lammps(deck, outdir, f'probe_{name}_{frac:.3f}', lmp)['pe'][-1]
            e_mc = calc.get_potential_energy(a)
            if abs(e_lmp - e_mc) > 1e-6 * max(abs(e_mc), sys_['eps']):
                raise SystemExit(f'{name}: LJ mismatch at r={r}: '
                                 f'mcpy {e_mc} vs LAMMPS {e_lmp}')
    print('LJ energies match LAMMPS for both systems')


def run_all(args):
    os.makedirs(args.outdir, exist_ok=True)
    probe_potential(args.outdir, args.lmp)
    liquid = [(name, s, mu) for name, s in SYSTEMS.items() for mu in s['mus']
              if is_liquid(s, mu)]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        list(ex.map(prepare_liquid, *zip(*liquid), [args.outdir] * len(liquid),
                    [args.lmp] * len(liquid)))
    keys = [(code, name, mu, rep) for name, s in SYSTEMS.items()
            for mu in s['mus'] for code in ('mcpy', 'lammps')
            for rep in range(REPLICAS)]
    # Longest jobs (dense argon mcpy) first so the pool tail stays short.
    keys.sort(key=lambda k: (k[0] != 'mcpy', k[1] != 'argon', -k[2]))
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        futures = {k: ex.submit(job, *k, args) for k in keys}
        means = {k: f.result() for k, f in futures.items()}
    with open(csv_path(args.outdir), 'w') as f:
        f.write('system,mu,mcpy_mean,mcpy_err,lammps_mean,lammps_err\n')
        for name, s in SYSTEMS.items():
            for mu in s['mus']:
                row = [name, f'{mu:.7g}']
                for code in ('mcpy', 'lammps'):
                    x = np.array([means[(code, name, mu, r)] for r in range(REPLICAS)])
                    row += [f'{x.mean():.6g}', f'{x.std(ddof=1) / math.sqrt(REPLICAS):.6g}']
                f.write(','.join(row) + '\n')


def csv_path(outdir):
    return os.path.join(outdir, 'lj_isotherms.csv')


def plot(outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = np.genfromtxt(csv_path(outdir), delimiter=',', names=True, dtype=None,
                         encoding=None)
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.4), sharex='col',
                             height_ratios=[2, 1], constrained_layout=True)
    for col, (name, title, xlabel) in enumerate((
            ('repulsive', r'(a) repulsive LJ fluid ($T^*=1$)', r'$\mu$ ($\epsilon$)'),
            ('argon', r'(b) argon ($T=87.79$ K)', r'$\mu$ (eV)'))):
        d = data[data['system'] == name]
        mu, m, sm, ml, sl = (d[k] for k in ('mu', 'mcpy_mean', 'mcpy_err',
                                            'lammps_mean', 'lammps_err'))
        top, bottom = axes[0, col], axes[1, col]
        top.errorbar(mu, ml, yerr=sl, fmt='o-', color='navy', lw=3, ms=8,
                     label='LAMMPS')
        top.errorbar(mu, m, yerr=sm, fmt='o-', color='crimson', lw=1.5, ms=4,
                     label='mcpy')
        top.set_ylabel(r'$\langle N \rangle$')
        top.set_title(title)
        top.legend()
        s = np.hypot(sm, sl)
        bottom.fill_between(mu, -2 * s, 2 * s, color='0.85', label=r'$\pm 2\sigma$')
        bottom.axhline(0.0, color='0.4', lw=0.8)
        bottom.errorbar(mu, m - ml, yerr=s, fmt='o', color='crimson', ms=4, capsize=3)
        bottom.set_xlabel(xlabel)
        bottom.set_ylabel(r'mcpy $-$ LAMMPS')
        bottom.legend(loc='lower left')
        z = (m - ml) / s
        print(f'{name}: max |d|/sigma = {np.abs(z).max():.2f}, '
              f'chi2 = {np.sum(z ** 2):.1f} over {len(z)} points')
    path = os.path.join(outdir, 'lj_isotherms.png')
    fig.savefig(path, dpi=300)
    print(f'plot: {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--lmp', default=LMP_DEFAULT)
    p.add_argument('--steps', type=int, default=200_000,
                   help='trial moves per run, the same for both codes')
    p.add_argument('--outdir', default='benchmark/lj_isotherms')
    p.add_argument('--plot', action='store_true', help='replot from the CSV only')
    args = p.parse_args()
    if not args.plot:
        run_all(args)
    plot(args.outdir)


if __name__ == '__main__':
    main()
