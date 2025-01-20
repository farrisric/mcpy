from ase.build import fcc111
import numpy as np

from mcpy.moves import DeletionMove, InsertionMove, DisplacementMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACECalculator


atoms = fcc111('Ag', size=(4, 4, 3), periodic=True, vacuum=8)
surface_indices = [a.index for a in atoms if a.tag == 1]
box = [atoms.cell[0], atoms.cell[1], np.array([0, 0, 6])]
z_shift = atoms[surface_indices[0]].position[2]-3

calculator = MACECalculator('/home/riccardo/Downloads/mace-large-density-agnesi-stress.model')
calculator.get_potential_energy(atoms)

species = ['Ag', 'O']

move_list = [[5, 5, 10],
             [DeletionMove(species=species,
                           seed=12,
                           operating_box=box,
                           z_shift=z_shift),
              InsertionMove(species=species,
                            seed=13,
                            operating_box=box,
                            min_max_insert=[1.5, 3.0],
                            z_shift=z_shift),
              DisplacementMove(species=species,
                               seed=14,
                               max_displacement=0.2)]]

move_selector = MoveSelector(*move_list)
T = 500
gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            calculator=calculator,
            mu={'Ag' : -2.8, 'O' : -4.9},
            units_type='metal',
            species=species,
            temperature=T,
            move_selector=move_selector,
            outfile=f'GCMC_{T}K.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=f'GCMC_{T}K.xyz')

gcmc.run(1000000)
