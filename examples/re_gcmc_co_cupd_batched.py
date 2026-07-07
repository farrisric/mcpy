"""CO adsorption phase diagram on a CuPd nanoparticle via batched-RE GCMC.

One replica per CO chemical potential (mu ladder), all replicas evaluated in
batched relaxed-energy passes on one GPU (AlchemiFCalculator). Moves: rigid CO
molecule insertion/deletion (mu = E(CO) + delta_mu, full molecular chemical
potential) plus Cu<->Pd permutation so the alloy can segregate under CO.

Outputs per replica outfiles/trajectories plus a coverage-vs-delta_mu phase
diagram (phase_diagram_co_cupd.png).

Run (alchemi env)::

    python examples/re_gcmc_co_cupd_batched.py --gcmc-steps 80
"""
import argparse
import os

import numpy as np
from ase.build import molecule
from ase.cluster import Octahedron

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import (MoleculeDeletionMove, MoleculeInsertionMove,  # noqa: E402
                        PermutationMove)
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.ensembles import BatchedReplicaExchange  # noqa: E402
from mcpy.calculators import AlchemiFCalculator  # noqa: E402
from mcpy.cell import SphericalCell  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--delta-mus', type=float, nargs='+',
                   default=[-1.5, -1.2, -0.9, -0.6, -0.3],
                   help='CO delta_mu ladder, one replica each (eV)')
    p.add_argument('--T', type=float, default=400.0)
    p.add_argument('--n-pd', type=int, default=5, help='Pd atoms in the Cu NP')
    p.add_argument('--gcmc-steps', type=int, default=80)
    p.add_argument('--exchange-interval', type=int, default=8)
    p.add_argument('--checkpoint', default='medium-mpa-0')
    p.add_argument('--rel-steps', type=int, default=30)
    p.add_argument('--rel-fmax', type=float, default=0.1)
    p.add_argument('--min-insert', type=float, default=1.3)
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--outdir', default='runs_co_cupd')
    return p.parse_args()


def build_nanoparticle(n_pd, seed):
    atoms = Octahedron('Cu', 4, cutoff=1)  # 38-atom truncated octahedron
    rng = np.random.default_rng(seed)
    pd_idx = rng.choice(len(atoms), size=n_pd, replace=False)
    symbols = np.array(atoms.get_chemical_symbols())
    symbols[pd_idx] = 'Pd'
    atoms.set_chemical_symbols(symbols.tolist())
    atoms.center(vacuum=10.0)
    atoms.pbc = False
    return atoms


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    n_rep = len(args.delta_mus)
    seeds = ss.generate_state(3 * n_rep + 2, dtype=np.uint32)
    move_seeds = [int(s) for s in seeds[:3 * n_rep]]
    master_seed, np_seed = int(seeds[-1]), int(seeds[-2])

    base_atoms = build_nanoparticle(args.n_pd, np_seed)
    n_metal = len(base_atoms)
    print(f'nanoparticle: {base_atoms.get_chemical_formula()} ({n_metal} atoms)')

    co = molecule('CO')

    calculator = AlchemiFCalculator(
        checkpoint=args.checkpoint,
        steps=args.rel_steps,
        fmax=args.rel_fmax,
        compile_model=not args.no_compile,
    )

    e_co = calculator.get_potential_energy(
        molecule('CO', cell=[20.0, 20.0, 20.0], pbc=False))
    print(f'E(CO, relaxed, isolated) = {e_co:.4f} eV')
    mus = [{'CO': e_co + d} for d in args.delta_mus]

    def gcmc_factory(mu, rank):
        atoms = base_atoms.copy()
        cell = SphericalCell(
            atoms, vacuum=3.5,
            species_radii={'Cu': 2.4, 'Pd': 2.5, 'C': 0.0, 'O': 0.0},
            seed=move_seeds[3 * rank] + 1,
        )
        s = move_seeds[3 * rank:3 * (rank + 1)]
        move_selector = MoveSelector(
            [2, 2, 1],
            [MoleculeInsertionMove(cell, co, 'CO', seed=s[0],
                                   min_insert=args.min_insert),
             MoleculeDeletionMove(cell, co, 'CO', seed=s[1]),
             PermutationMove(species=['Cu', 'Pd'], seed=s[2])],
            n_moves=3,
        )
        d = args.delta_mus[rank]
        tag = f'{base_atoms.get_chemical_formula()}_CO_dmu_{d}'
        return GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[cell],
            calculator=calculator,
            mu=mu,
            units_type='metal',
            species=['Cu', 'Pd'],
            molecules={'CO': co},
            temperature=args.T,
            move_selector=move_selector,
            random_seed=s[0] + 17,
            outfile=os.path.join(args.outdir, f'gcmc_{tag}.out'),
            traj_file=os.path.join(args.outdir, f'gcmc_{tag}.xyz'),
            outfile_write_interval=1,
            trajectory_write_interval=1,
        )

    pt = BatchedReplicaExchange(
        gcmc_factory,
        calculator=calculator,
        mus=mus,
        gcmc_steps=args.gcmc_steps,
        exchange_interval=args.exchange_interval,
        write_out_interval=args.exchange_interval,
        seed=master_seed,
        outfile=os.path.join(args.outdir, 'replica_exchange_co_cupd.log'),
        global_minimum_file=os.path.join(args.outdir, 'global_minimum.xyz'),
    )
    pt.run()

    plot_phase_diagram(args, n_metal)
    render_snapshots(args, base_atoms.get_chemical_formula())


def plot_phase_diagram(args, n_metal):
    """Coverage vs delta_mu from the per-replica outfiles (last 50%)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cov, err = [], []
    for d in args.delta_mus:
        n_series = []
        path = None
        for f in os.listdir(args.outdir):
            if f.endswith(f'_dmu_{d}.out'):
                path = os.path.join(args.outdir, f)
        with open(path) as fh:
            for line in fh:
                s = line.split()
                if s and s[0].isdigit():
                    n_series.append((int(s[1]) - n_metal) / 2.0)
        tail = np.array(n_series[len(n_series) // 2:])
        nb = max(2, min(5, len(tail)))
        blocks = [b.mean() for b in np.array_split(tail, nb) if len(b)]
        cov.append(np.mean(blocks))
        err.append(np.std(blocks, ddof=1) / np.sqrt(len(blocks)))

    fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
    ax.errorbar(args.delta_mus, cov, yerr=err, fmt='o-', capsize=4)
    ax.set_xlabel(r'$\Delta\mu_\mathrm{CO}$ (eV)')
    ax.set_ylabel(r'$\langle N_\mathrm{CO} \rangle$ adsorbed')
    ax.set_title(f'CO on CuPd nanoparticle, T = {args.T:.0f} K')
    out = os.path.join(args.outdir, 'phase_diagram_co_cupd.png')
    fig.savefig(out, dpi=150)
    print(f'phase diagram: {out}')
    print('coverage:', {d: f'{c:.1f}±{e:.1f}' for d, c, e in
                        zip(args.delta_mus, cov, err)})


def render_snapshots(args, formula):
    """One rendered snapshot per replica (final trajectory frame) plus the
    cross-replica global minimum, on a single row."""
    import ase.io
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms

    n = len(args.delta_mus)
    fig, axes = plt.subplots(1, n + 1, figsize=(2.6 * (n + 1), 3.0),
                             constrained_layout=True)
    for ax, d in zip(axes[:n], args.delta_mus):
        atoms = ase.io.read(
            os.path.join(args.outdir, f'gcmc_{formula}_CO_dmu_{d}.xyz'),
            index=-1)
        n_co = (len(atoms) - len([s for s in atoms.get_chemical_symbols()
                                  if s in ('Cu', 'Pd')])) // 2
        plot_atoms(atoms, ax, rotation='12z,-75x', show_unit_cell=0)
        ax.set_title(f'$\\Delta\\mu$ = {d} eV\n{n_co} CO', fontsize=9)
        ax.set_axis_off()
    gm = ase.io.read(os.path.join(args.outdir, 'global_minimum.xyz'))
    plot_atoms(gm, axes[n], rotation='12z,-75x', show_unit_cell=0)
    axes[n].set_title(f'global minimum\n{gm.get_chemical_formula()}', fontsize=9)
    axes[n].set_axis_off()
    out = os.path.join(args.outdir, 'snapshots_co_cupd.png')
    fig.savefig(out, dpi=180)
    print(f'snapshots: {out}')


if __name__ == '__main__':
    main()
