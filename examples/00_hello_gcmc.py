"""Smallest possible GCMC run: a Lennard-Jones fluid in a periodic box.

Runs on a laptop CPU in a few seconds. No GPU, no MACE, no downloads.
This is the textbook GCMC test case, and the same system mcpy is validated
against LAMMPS ``fix gcmc`` with (see ``benchmark/README.md``), so the average
particle number it settles at is a real, checkable number.

    python examples/00_hello_gcmc.py

Writes ``hello_gcmc.out`` (step / N / energy / acceptance ratios) and
``hello_gcmc.xyz`` (trajectory).

Once this makes sense, read ``examples/03_gcmc_surface_mace.py`` for the same loop
driven by a machine-learned potential on a real surface.
"""
from ase import Atoms
from ase.calculators.lj import LennardJones

from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.moves import DeletionMove, DisplacementMove, InsertionMove, MoveSelector
from mcpy.utils.logging import configure as configure_logging

# LJ reduced units: sigma = epsilon = 1. mcpy's 'LJ' unit system sets the
# de Broglie wavelength to 1, so mu is the reduced chemical potential.
BOX = 9.0          # box side, in sigma
CUTOFF = 3.0       # pair cutoff, in sigma
TEMPERATURE = 2.0  # reduced temperature
MU = -4.0          # reduced chemical potential of the reservoir
STEPS = 3000

# One seed per stochastic component. Leaving any of them at None makes the
# run unrepeatable.
SEED_INS, SEED_DEL, SEED_DIS, SEED_SEL, SEED_ENS = 1, 2, 3, 4, 5


def main():
    configure_logging()

    # 1. An empty periodic box. GCMC fills it from the reservoir.
    atoms = Atoms(cell=[BOX, BOX, BOX], pbc=True)

    # 2. The region where particles may be inserted or deleted. ``Cell`` is
    #    the whole periodic box.
    cell = Cell(atoms)

    # 3. The calculator. Any ASE calculator works directly; wrap it in
    #    ``mcpy.calculators.BaseCalculator`` instead if each trial structure
    #    should be relaxed before its energy is used.
    calculator = LennardJones(sigma=1.0, epsilon=1.0, rc=CUTOFF, smooth=False)

    # 4. The trial moves, sampled with equal weight. Insertion and deletion
    #    change N; displacement equilibrates the particles already present.
    move_selector = MoveSelector(
        [1, 1, 1],
        [InsertionMove(cell, species=['H'], seed=SEED_INS),
         DeletionMove(cell, species=['H'], seed=SEED_DEL),
         DisplacementMove(species=['H'], max_displacement=0.3, seed=SEED_DIS)],
        seed=SEED_SEL,
    )

    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[cell],
        calculator=calculator,
        mu={'H': MU},
        species=['H'],
        units_type='LJ',
        temperature=TEMPERATURE,
        move_selector=move_selector,
        random_seed=SEED_ENS,
        outfile='hello_gcmc.out',
        traj_file='hello_gcmc.xyz',
        outfile_write_interval=100,
        trajectory_write_interval=100,
    )
    gcmc.run(STEPS)

    # The observable of a GCMC run is an average over the equilibrated part
    # of the trajectory, never the last frame.
    print(f'final N = {len(gcmc.atoms)}, final E = {gcmc.E_old:.3f}')


if __name__ == '__main__':
    main()
