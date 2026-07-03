"""GCMC insertion/deletion of a rigid O2 molecule in a periodic box (LJ units).

Runs without torch/mace: uses ASE's LennardJones calculator directly.
Demonstrates the molecular-move API; see
docs/gcmc_acceptance_convention.rst for the acceptance convention
(mu is the full molecular chemical potential).
"""
import argparse

from ase import Atoms
from ase.calculators.lj import LennardJones

from mcpy.utils.logging import configure as configure_logging

configure_logging()

from mcpy.moves import (MoleculeDeletionMove, MoleculeInsertionMove,  # noqa: E402
                        MoveSelector)
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble  # noqa: E402
from mcpy.cell import Cell  # noqa: E402


class LJEnergy:
    """Minimal calculator adapter: plain LJ energy, no relaxation."""

    def __init__(self):
        self.lj = LennardJones(sigma=1.0, epsilon=1.0, rc=3.0)

    def get_potential_energy(self, atoms):
        atoms.calc = self.lj
        return atoms.get_potential_energy()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--box', type=float, default=8.0, help='Cubic box edge')
    p.add_argument('--T', type=float, default=1.0, help='Temperature (LJ units)')
    p.add_argument('--mu', type=float, default=-2.0,
                   help='Full molecular chemical potential of O2 (LJ units)')
    p.add_argument('--bond', type=float, default=1.1, help='O-O bond length')
    p.add_argument('--min-insert', type=float, default=0.9,
                   help='Min insertion distance to existing atoms')
    p.add_argument('--gcmc-steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    o2 = Atoms('O2', positions=[[0, 0, 0], [0, 0, args.bond]])
    atoms = Atoms(cell=[args.box] * 3, pbc=True)
    cell = Cell(atoms, species_radii={'O': 0.0}, seed=args.seed)

    move_selector = MoveSelector(
        [1, 1],
        [MoleculeInsertionMove(cell, o2, 'O2', seed=args.seed + 1,
                               min_insert=args.min_insert),
         MoleculeDeletionMove(cell, o2, 'O2', seed=args.seed + 2)],
        seed=args.seed + 3,
    )

    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[cell],
        units_type='LJ',
        calculator=LJEnergy(),
        mu={'O2': args.mu},
        species=[],
        temperature=args.T,
        move_selector=move_selector,
        molecules={'O2': o2},
        random_seed=args.seed,
        outfile='gcmc_molecule.out',
        traj_file='gcmc_molecule.xyz',
    )
    gcmc.run(args.gcmc_steps)


if __name__ == '__main__':
    main()
