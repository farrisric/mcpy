"""GCMC of metal + gas adsorption on a generic fcc surface with two CustomCell regions.

Supports fcc(111), fcc(100), fcc(110), fcc(211) terminations.
"""
import argparse
import os

import numpy as np
from ase.build import fcc100, fcc111
from ase.constraints import FixAtoms

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import MACE_F_Calculator  # noqa: E402
from mcpy.cell import CustomCell as Cell  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-path', required=True, help='Path to the MACE model file')
    p.add_argument('--delta-mu-gas', type=float, required=True, help='Shift applied to mu_gas (eV)')
    p.add_argument('--device', default='cuda', help='Torch device for MACE')
    p.add_argument('--metal', default='Au', help='Metal species symbol')
    p.add_argument('--gas', default='O', help='Gas species symbol')
    p.add_argument('--lattice-param', type=float, default=4.1755, help='fcc lattice constant (Å)')
    p.add_argument('--surface-type', default='111', choices=['111', '100', '110', '211'])
    p.add_argument('--surface-size', type=int, nargs=3, default=[4, 4, 3])
    p.add_argument('--vacuum', type=float, default=8.0, help='Vacuum spacing (Å)')
    p.add_argument('--T', type=float, default=500.0, help='Temperature (K)')
    p.add_argument('--mu-metal', type=float, default=-3.26999, help='Bulk metal DFT energy (eV)')
    p.add_argument('--delta-mu-metal', type=float, default=-0.16, help='Shift to mu_metal (eV)')
    p.add_argument('--mu-gas', type=float, default=-4.91, help='Reference mu_gas (eV)')
    p.add_argument('--moves-per-step', type=int, default=25,
                   help='Insertion+deletion moves per species per GCMC step')
    p.add_argument('--gcmc-steps', type=int, default=200, help='Number of GCMC steps')
    p.add_argument('--write-interval', type=int, default=1, help='Outfile/traj write interval')
    p.add_argument('--min-insert', type=float, default=0.5,
                   help='Min insertion distance to existing atoms (Å)')
    p.add_argument('--cell-height', type=float, default=5.0, help='Insertion cell height (Å)')
    p.add_argument('--cell-bottom-z', type=float, default=10.411, help='Cell bottom z (Å)')
    p.add_argument('--metal-radius', type=float, default=2.75)
    p.add_argument('--gas-radius', type=float, default=2.11)
    p.add_argument('--rel-max-steps', type=int, default=40, help='LBFGS relaxation steps')
    p.add_argument('--rel-fmax', type=float, default=0.1, help='LBFGS force convergence (eV/Å)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--outdir', default='.', help='Output directory')
    return p.parse_args()


def build_slab(args):
    size = tuple(args.surface_size)
    if args.surface_type == '111':
        atoms = fcc111(args.metal, a=args.lattice_param, size=size,
                       periodic=True, vacuum=args.vacuum)
        bottom = [a.index for a in atoms if a.tag == size[-1]]
    elif args.surface_type == '100':
        atoms = fcc100(args.metal, a=args.lattice_param, size=size,
                       periodic=True, vacuum=args.vacuum)
        bottom = [a.index for a in atoms if a.tag == size[-1]]
    elif args.surface_type == '110':
        # 110 termination: fix last two layers
        atoms = fcc100(args.metal, a=args.lattice_param, size=size,
                       periodic=True, vacuum=args.vacuum)
        bottom = [a.index for a in atoms if a.tag in (size[-1], size[-1] - 1)]
    elif args.surface_type == '211':
        atoms = fcc100(args.metal, a=args.lattice_param, size=size,
                       periodic=True, vacuum=args.vacuum)
        layer = size[0] * size[1]
        bottom = list(range(len(atoms)))[-layer:]
    else:
        raise ValueError(f'Invalid surface type: {args.surface_type}')
    atoms.set_constraint(FixAtoms(indices=bottom))
    return atoms


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seeds = [int(s) for s in ss.generate_state(4, dtype=np.uint32)]

    atoms = build_slab(args)

    metal_radii = {args.metal: args.metal_radius, args.gas: 0.0}
    gas_radii = {args.metal: args.gas_radius, args.gas: 0.0}

    cell_metal = Cell(atoms, custom_height=args.cell_height, bottom_z=args.cell_bottom_z,
                      species_radii=metal_radii)
    cell_gas = Cell(atoms, custom_height=args.cell_height, bottom_z=args.cell_bottom_z,
                    species_radii=gas_radii)

    calculator = MACE_F_Calculator(
        model_paths=args.model_path,
        steps=args.rel_max_steps,
        fmax=args.rel_fmax,
        cueq=False,
        device=args.device,
    )

    n = args.moves_per_step
    move_selector = MoveSelector(
        [n, n, n, n],
        [DeletionMove(cell_metal, species=[args.metal], seed=seeds[0]),
         DeletionMove(cell_gas, species=[args.metal], seed=seeds[1]),
         InsertionMove(cell_metal, species=[args.metal], min_insert=args.min_insert, seed=seeds[2]),
         InsertionMove(cell_gas, species=[args.gas], min_insert=args.min_insert, seed=seeds[3])],
    )

    mus = {
        args.metal: args.mu_metal + args.delta_mu_metal,
        args.gas: args.mu_gas + args.delta_mu_gas,
    }

    tag = f'{atoms.get_chemical_formula()}_{args.gas}_dmu_{args.delta_mu_gas}'
    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[cell_metal, cell_gas],
        calculator=calculator,
        mu=mus,
        units_type='metal',
        species=[args.metal, args.gas],
        temperature=args.T,
        move_selector=move_selector,
        outfile=os.path.join(args.outdir, f'gcmc_relax_{tag}.out'),
        trajectory_write_interval=args.write_interval,
        outfile_write_interval=args.write_interval,
        traj_file=os.path.join(args.outdir, f'gcmc_relax_{tag}.xyz'),
    )

    gcmc.run(args.gcmc_steps)


if __name__ == '__main__':
    main()
