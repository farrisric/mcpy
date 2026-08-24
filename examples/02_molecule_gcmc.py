"""Molecular GCMC: rigid Lennard-Jones dimers exchanged with a reservoir.

Same box and potential as ``00_hello_gcmc.py``, but the exchanged species is
a whole molecule instead of a single atom. This is the setup mcpy's molecular
moves are validated against LAMMPS ``fix gcmc`` with (``benchmark/README.md``).

Three molecular moves appear here:
  * ``MoleculeInsertionMove`` places the rigid template at a random position
    with a random orientation,
  * ``MoleculeDeletionMove`` removes a whole molecule at once,
  * ``MoleculeDisplacementMove`` translates and rotates one molecule rigidly.

Membership is tracked by a per-atom ``molecule_id`` array that survives
rejection and is written into the trajectory, so a run can restart from its
own ``.xyz`` with molecules intact.

    python examples/02_molecule_gcmc.py

Writes ``molecule_gcmc.out`` and ``molecule_gcmc.xyz``. For a real adsorbate
(CO, H2O, O2) on a surface, swap the template for an ASE-built molecule and
the calculator for a machine-learned potential; see ``docs/molecular_adsorbates.rst``.
"""
from ase import Atoms
from ase.calculators.lj import LennardJones

from mcpy.cell import Cell
from mcpy.ensembles import GrandCanonicalEnsemble
from mcpy.moves import (MoleculeDeletionMove, MoleculeDisplacementMove,
                        MoleculeInsertionMove, MoveSelector)
from mcpy.utils.logging import configure as configure_logging

BOX = 9.0          # box side, in sigma
CUTOFF = 3.0       # pair cutoff, in sigma
TEMPERATURE = 2.0  # reduced temperature
# mu is the chemical potential of the WHOLE molecule: the energy of one
# molecule in the reservoir plus its translational, rotational and pressure
# contributions. It is not the per-atom value.
MU = -6.0
STEPS = 1000

SEED_INS, SEED_DEL, SEED_DIS, SEED_SEL, SEED_ENS = 1, 2, 3, 4, 5


def main():
    configure_logging()

    atoms = Atoms(cell=[BOX, BOX, BOX], pbc=True)
    cell = Cell(atoms)
    calculator = LennardJones(sigma=1.0, epsilon=1.0, rc=CUTOFF, smooth=False)

    # The template is centered on its center of mass when the move is built;
    # its name ('H2') is the key used in ``mu`` and in ``molecules``.
    template = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])

    move_selector = MoveSelector(
        [1, 1, 1],
        [MoleculeInsertionMove(cell, template, 'H2', seed=SEED_INS),
         MoleculeDeletionMove(cell, template, 'H2', seed=SEED_DEL),
         MoleculeDisplacementMove(cell, template, 'H2', max_displacement=0.3,
                                  seed=SEED_DIS)],
        seed=SEED_SEL,
    )

    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[cell],
        calculator=calculator,
        mu={'H2': MU},
        # ``species`` lists ATOMIC exchangeable species only; leave it empty
        # when every exchanged species is molecular.
        species=[],
        molecules={'H2': template},
        units_type='LJ',
        temperature=TEMPERATURE,
        move_selector=move_selector,
        random_seed=SEED_ENS,
        outfile='molecule_gcmc.out',
        traj_file='molecule_gcmc.xyz',
        outfile_write_interval=100,
        trajectory_write_interval=100,
    )
    gcmc.run(STEPS)

    print(f'final: {len(gcmc.atoms) // len(template)} molecules, '
          f'E = {gcmc.E_old:.3f}')


if __name__ == '__main__':
    main()
