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

from mcpy.moves import (MoleculeDeletionMove, MoleculeDisplacementMove,  # noqa: E402
                        MoleculeInsertionMove, PermutationMove)
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
    p.add_argument('--octa-length', type=int, default=4,
                   help='Octahedron length (4,1 -> 38 atoms; 9,3 -> 405)')
    p.add_argument('--octa-cutoff', type=int, default=1)
    p.add_argument('--gcmc-steps', type=int, default=80,
                   help='GCMC steps per replica')
    p.add_argument('--exchange-interval', type=int, default=8,
                   help='Steps between replica-exchange attempts; use >= 50 '
                        'for per-mu coverage curves (small values over-mix '
                        'the ladder)')
    p.add_argument('--checkpoint', default='medium-mpa-0')
    p.add_argument('--rel-steps', type=int, default=30)
    p.add_argument('--rel-fmax', type=float, default=0.1)
    p.add_argument('--min-insert', type=float, default=1.3)
    p.add_argument('--mol-disp-weight', type=int, default=0,
                   help='Weight of the rigid CO translate+rotate move '
                        '(0 = disabled, matching pre-move behaviour)')
    p.add_argument('--mol-disp-max', type=float, default=1.0,
                   help='Max COM displacement of the rigid move (A)')
    p.add_argument('--mol-disp-angle', type=float, default=None,
                   help='Max rotation angle of the rigid move (rad); '
                        'None = full uniform rotation')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--seed', type=int, default=7,
                   help='Master seed for moves, RE, and the Pd placement')
    p.add_argument('--outdir',
                   default=os.path.expanduser('~/mcpy_tmp_runs/co_cupd'),
                   help='Output directory (kept outside the git repo)')
    p.add_argument('--init-dir', default=None,
                   help='Seed each replica from the last trajectory frame of a '
                        'previous run in this directory (same delta-mus) to '
                        'continue toward convergence')
    return p.parse_args()


def build_nanoparticle(n_pd, seed, length=4, cutoff=1):
    atoms = Octahedron('Cu', length, cutoff=cutoff)
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

    base_atoms = build_nanoparticle(args.n_pd, np_seed,
                                    args.octa_length, args.octa_cutoff)
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
        if args.init_dir:
            import ase.io
            d = args.delta_mus[rank]
            prev = os.path.join(
                args.init_dir,
                f'gcmc_{base_atoms.get_chemical_formula()}_CO_dmu_{d}.xyz')
            atoms = ase.io.read(prev, index=-1)  # molecule_id round-trips
            atoms.pbc = False
        cell = SphericalCell(
            atoms, vacuum=3.5,
            species_radii={'Cu': 2.4, 'Pd': 2.5, 'C': 0.0, 'O': 0.0},
            seed=move_seeds[3 * rank] + 1,
        )
        s = move_seeds[3 * rank:3 * (rank + 1)]
        weights = [2, 2, 1]
        moves = [MoleculeInsertionMove(cell, co, 'CO', seed=s[0],
                                       min_insert=args.min_insert),
                 MoleculeDeletionMove(cell, co, 'CO', seed=s[1]),
                 PermutationMove(species=['Cu', 'Pd'], seed=s[2])]
        if args.mol_disp_weight > 0:
            weights.append(args.mol_disp_weight)
            moves.append(MoleculeDisplacementMove(
                cell, co, 'CO', seed=s[0] + 101,
                max_displacement=args.mol_disp_max,
                max_angle=args.mol_disp_angle))
        move_selector = MoveSelector(weights, moves, n_moves=3)
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

    plot_coverage(args, n_metal)
    render_snapshots(args, base_atoms.get_chemical_formula())
    try:
        library_phase_diagram(args, base_atoms.get_chemical_formula(), e_co)
    except ValueError as exc:
        # Deep continuation chains may hold no adsorbate-free reference frame;
        # pool the chain's trajectories by hand for the final diagram then.
        print(f'library phase diagram skipped: {exc}')


def library_phase_diagram(args, formula, e_co):
    """The proper thermodynamic phase diagram via mcpy's own analysis:
    per-stoichiometry free-energy lines, stable envelope, transitions, and
    per-phase structure thumbnails. CO molecules are counted via their single
    C atom; mu_ref = E(CO) matches the full-molecular-mu convention."""
    import ase.io
    from mcpy.utils.phase_diagram import plot_phase_diagram

    frames = []
    for d in args.delta_mus:
        frames.append(ase.io.read(
            os.path.join(args.outdir, f'gcmc_{formula}_CO_dmu_{d}.xyz'),
            index=':'))
        # Continuation runs start CO-loaded; pull the adsorbate-free
        # reference frames from the run they were seeded from.
        if args.init_dir:
            frames.append(ase.io.read(
                os.path.join(args.init_dir, f'gcmc_{formula}_CO_dmu_{d}.xyz'),
                index=':'))
    pad = 0.15
    out = os.path.join(args.outdir, 'phase_diagram_co_cupd.png')
    res = plot_phase_diagram(
        frames,
        adsorbate='C',
        metal_symbols=('Cu', 'Pd'),
        mu_ref=e_co,
        kind='nano',
        T=args.T,
        dmu_range=(min(args.delta_mus) - pad, max(args.delta_mus) + pad),
        rotation='12z,-75x',
        gamma_in_ev=True,
        outfile=out,
        adsorbate_label='CO',
        atoms_per_reservoir_molecule=1,
    )
    print(f'phase diagram (library): {out}')
    print('stable CO stoichiometries:',
          [res['stoich'][i] for i in res['phase_order']])
    print('transitions at delta_mu:', np.round(res['transitions'], 3).tolist())
    return res


def plot_coverage(args, n_metal):
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
        # Stationarity check: first vs second half of the post-burn tail must
        # agree within 2x the combined block error, else more steps needed.
        h1, h2 = np.array_split(tail, 2)
        s_comb = np.hypot(h1.std(ddof=1) / np.sqrt(max(len(h1), 2)),
                          h2.std(ddof=1) / np.sqrt(max(len(h2), 2)))
        drift = abs(h1.mean() - h2.mean())
        status = 'CONVERGED' if drift < 2 * max(s_comb, 1e-9) else 'NOT CONVERGED'
        print(f'  dmu={d}: tail halves {h1.mean():.1f} vs {h2.mean():.1f} '
              f'(drift {drift:.1f}, 2s={2 * s_comb:.1f}) {status}')

    fig, ax = plt.subplots(figsize=(5, 3.4), constrained_layout=True)
    ax.errorbar(args.delta_mus, cov, yerr=err, fmt='o-', capsize=4)
    ax.set_xlabel(r'$\Delta\mu_\mathrm{CO}$ (eV)')
    ax.set_ylabel(r'$\langle N_\mathrm{CO} \rangle$ adsorbed')
    ax.set_title(f'CO on CuPd nanoparticle, T = {args.T:.0f} K')
    out = os.path.join(args.outdir, 'coverage_co_cupd.png')
    fig.savefig(out, dpi=150)
    print(f'coverage isotherm: {out}')
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
