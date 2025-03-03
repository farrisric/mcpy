from ase.build import fcc111
from ase.constraints import FixAtoms
import numpy as np

from mcpy.moves import DeletionMove, InsertionMove, DisplacementMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.utils.utils import get_volume

atoms = fcc111('Ag', size=(4, 4, 3), periodic=True, vacuum=8)
bottom_layer = [a.index for a in atoms if a.tag == 3]
constraint = FixAtoms(indices=bottom_layer)
atoms.set_constraint(constraint)

surface_indices = [a.index for a in atoms if a.tag == 1]
box = [atoms.cell[0], atoms.cell[1], np.array([0, 0, 6])]
z_shift = atoms[surface_indices[0]].position[2]-3

r_min = 1.3
volume = get_volume(box)
volume -= - 4/3 * np.pi * (r_min**3) * len(atoms)
print(volume)

calculator = MACE_F_Calculator(
                model_paths='/home/riccardo/Downloads/mace-large-density-agnesi-stress.model',
                steps=20,
                fmax=0.1
                )

species = ['Ag', 'O']

move_list = [[1,1],
             [DeletionMove(species=species,
                           seed=12,
                           operating_box=box,
                           z_shift=z_shift),
              InsertionMove(species=species,
                            seed=13,
                            operating_box=box,
                            min_max_insert=[1.5, 3.0],
                            z_shift=z_shift),
            #   DisplacementMove(species=species,
            #                    seed=14,
            #                    max_displacement=0.2)]
            ]]

move_selector = MoveSelector(*move_list)
T = 500
gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            calculator=calculator,
            mu={'Ag' : -2.8, 'O' : -4.9},
            units_type='metal',
            species=species,
            species_volume=['Ag'],
            volume=volume,
            temperature=T,
            move_selector=move_selector,
            outfile=f'GCMC_{T}K.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=f'GCMC_{T}K.xyz')

gcmc.run(1000000)
