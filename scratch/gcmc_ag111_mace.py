"""GCMC of O adsorption on an Ag(111) surface (~1000 atoms) using ASE MACE (MACE-MP).

Same setup as gcmc_ag111_alchemi.py but uses the ASE MACE calculator (LBFGS relaxation).

Typical usage:
  python gcmc_ag111_mace.py --device cuda
  python gcmc_ag111_mace.py --device cuda --delta-mu-O -1.0
"""
import argparse
import logging
import os

import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms
from mace.calculators import mace_mp

from mcpy.utils.logging import configure as configure_logging

configure_logging()
logger = logging.getLogger('mcpy')

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import MACE_F_Calculator  # noqa: E402
from mcpy.cell import CustomCell  # noqa: E402

AG_LATTICE = 4.086


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='medium-mpa-0',
                   help='MACE-MP model name or path to .model file')
    p.add_argument('--device', default='cuda', help='Torch device (cuda or cpu)')
    p.add_argument('--fmax', type=float, default=0.05, help='LBFGS force threshold (eV/Å)')
    p.add_argument('--relax-steps', type=int, default=300, help='Max LBFGS steps per eval')
    p.add_argument('--T', type=float, default=500.0, help='Temperature (K)')
    p.add_argument('--steps', type=int, default=5_000_000, help='Number of GCMC steps')
    p.add_argument('--mu-Ag', type=float, default=-2.99, help='Chemical potential of Ag (eV)')
    p.add_argument('--mu-O', type=float, default=-4.91, help='Reference mu_O (eV)')
    p.add_argument('--delta-mu-O', type=float, default=-0.3, help='Shift applied to mu_O (eV)')
    p.add_argument('--slab-size', type=int, nargs=3, default=[16, 16, 4],
                   help='fcc111 supercell (Nx Ny Nlayers); default 16 16 4 = 1024 atoms')
    p.add_argument('--vacuum', type=float, default=10.0, help='Vacuum above/below slab (Å)')
    p.add_argument('--cell-height', type=float, default=5.0, help='O insertion region height (Å)')
    p.add_argument('--fix-layers', type=int, default=2, help='Number of bottom layers to freeze')
    p.add_argument('--min-insert', type=float, default=1.5,
                   help='Min insertion distance to existing atoms (Å)')
    p.add_argument('--seed', type=int, default=None, help='Master seed (random if unset)')
    p.add_argument('--write-interval', type=int, default=10,
                   help='Outfile/trajectory write interval')
    p.add_argument('--outdir', default='.', help='Output directory')
    return p.parse_args()


def build_slab(args):
    nx, ny, nlayers = args.slab_size
    atoms = fcc111('Ag', a=AG_LATTICE, size=(nx, ny, nlayers),
                   periodic=True, vacuum=args.vacuum)
    fix_tag_threshold = nlayers - args.fix_layers + 1
    bottom = [a.index for a in atoms if a.tag >= fix_tag_threshold]
    atoms.set_constraint(FixAtoms(indices=bottom))
    return atoms


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))

    atoms = build_slab(args)

    z_top = float(atoms.positions[atoms.get_tags() == 1, 2].max())

    species_radii = {'Ag': 1.44, 'O': 0.0}
    cell = CustomCell(atoms, custom_height=args.cell_height, bottom_z=z_top,
                      species_radii=species_radii)

    ase_calc = mace_mp(model=args.checkpoint, device=args.device, default_dtype='float32')
    calculator = MACE_F_Calculator(model_paths=ase_calc, steps=args.relax_steps, fmax=args.fmax,
                                   device=args.device)

    logger.info('Calculator: ASE MACE-MP  |  model=%s  |  device=%s', args.checkpoint, args.device)

    species = ['O']
    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(cell, species=species, seed=seed_del),
         InsertionMove(cell, species=species, min_insert=args.min_insert, seed=seed_ins)],
    )

    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}

    nx, ny, nlayers = args.slab_size
    tag = f'Ag{nx * ny * nlayers}_111_dmuO_{args.delta_mu_O}_mace_lbfgs'

    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[cell],
        calculator=calculator,
        mu=mus,
        units_type='metal',
        species=species,
        temperature=args.T,
        move_selector=move_selector,
        outfile=os.path.join(args.outdir, f'gcmc_{tag}.out'),
        traj_file=os.path.join(args.outdir, f'gcmc_{tag}.xyz'),
        trajectory_write_interval=args.write_interval,
        outfile_write_interval=args.write_interval,
    )

    gcmc.run(args.steps)


if __name__ == '__main__':
    main()
