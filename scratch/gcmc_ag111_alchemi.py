"""GCMC of O adsorption on an Ag(111) surface (~1000 atoms) using the nvalchemi GPU stack.

Builds a 16×16×4 Ag(111) slab (1024 atoms) by default, fixes the bottom two layers,
and runs GCMC inserting/deleting O in a 5 Å region directly above the surface.

Requirements:
  pip install 'nvalchemi-toolkit[mace]'

Typical usage:
  # FIRE relaxation per step (recommended, GPU-native so fast enough):
  python gcmc_ag111_alchemi.py --checkpoint medium-mpa-0 --device cuda

  # Energy-only (fastest, no geometry relaxation):
  python gcmc_ag111_alchemi.py --checkpoint medium-mpa-0 --no-relax

  # Adjust chemical potential sweep:
  python gcmc_ag111_alchemi.py --checkpoint medium-mpa-0 --delta-mu-O -1.0
"""
import argparse
import os

import numpy as np
from ase.build import fcc111
from ase.constraints import FixAtoms

import logging

from mcpy.utils.logging import configure as configure_logging

configure_logging()
logger = logging.getLogger('mcpy')

from mcpy.moves import DeletionMove, InsertionMove  # noqa: E402
from mcpy.moves.move_selector import MoveSelector  # noqa: E402
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.calculators import AlchemiCalculator, AlchemiFCalculator  # noqa: E402
from mcpy.cell import CustomCell  # noqa: E402

AG_LATTICE = 4.086  # Ag fcc lattice constant (Angstrom)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', default='medium-mpa-0',
                   help='Named checkpoint (e.g. medium-mpa-0) or path to a .pt file')
    p.add_argument('--no-relax', action='store_true',
                   help='Use AlchemiCalculator (energy-only); default uses AlchemiFCalculator')
    p.add_argument('--device', default='cuda', help='Torch device (cuda or cpu)')
    p.add_argument('--no-cueq', action='store_true', help='Disable cuEquivariance kernel fusion')
    p.add_argument('--no-compile', action='store_true', help='Disable torch.compile')
    p.add_argument('--fmax', type=float, default=0.05, help='FIRE force threshold (eV/Å)')
    p.add_argument('--relax-steps', type=int, default=300, help='Max FIRE steps per eval')
    p.add_argument('--optimizer', default='fire2', choices=['fire', 'fire2'],
                   help='FIRE variant (fire2 typically converges in fewer steps)')
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
    # tag=1 is top layer, tag=nlayers is bottom layer (ASE convention)
    fix_tag_threshold = nlayers - args.fix_layers + 1
    bottom = [a.index for a in atoms if a.tag >= fix_tag_threshold]
    atoms.set_constraint(FixAtoms(indices=bottom))
    return atoms


def build_calculator(args):
    enable_cueq = not args.no_cueq
    compile_model = not args.no_compile
    if args.no_relax:
        return AlchemiCalculator(
            checkpoint=args.checkpoint,
            device=args.device,
            enable_cueq=enable_cueq,
            compile_model=compile_model,
        )
    return AlchemiFCalculator(
        checkpoint=args.checkpoint,
        steps=args.relax_steps,
        fmax=args.fmax,
        device=args.device,
        enable_cueq=enable_cueq,
        compile_model=compile_model,
        optimizer=args.optimizer,
    )


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ss = np.random.SeedSequence(args.seed)
    seed_del, seed_ins = (int(s) for s in ss.generate_state(2, dtype=np.uint32))

    atoms = build_slab(args)

    # Insertion cell starts at the top surface layer and extends upward
    z_top = float(atoms.positions[atoms.get_tags() == 1, 2].max())
    cell_bottom_z = z_top

    # Ag atomic radius for free-volume exclusion; O=0 so inserted O doesn't self-exclude
    species_radii = {'Ag': 1.44, 'O': 0.0}

    cell = CustomCell(
        atoms,
        custom_height=args.cell_height,
        bottom_z=cell_bottom_z,
        species_radii=species_radii,
    )

    calculator = build_calculator(args)

    import torch
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'
    logger.info('Calculator device: %s  |  CUDA available: %s  |  GPU: %s',
                args.device, torch.cuda.is_available(), gpu_name)

    species = ['O']
    move_selector = MoveSelector(
        [1, 1],
        [DeletionMove(cell, species=species, seed=seed_del),
         InsertionMove(cell, species=species, min_insert=args.min_insert, seed=seed_ins)],
    )

    mus = {'Ag': args.mu_Ag, 'O': args.mu_O + args.delta_mu_O}

    nx, ny, nlayers = args.slab_size
    mode = 'nrelax' if args.no_relax else f'fire_fmax{args.fmax}'
    tag = f'Ag{nx * ny * nlayers}_111_dmuO_{args.delta_mu_O}_{mode}'

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
