from ase.build import fcc111
from ase.constraints import FixAtoms
import numpy as np
from mcpy.moves import DeletionMove
from mcpy.moves import InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACECalculator
from mcpy.calculators import MACE_F_Calculator
from mcpy.utils.utils import get_volume

atoms = fcc111('Ag', size=(4, 4, 3), periodic=True, vacuum=8)
bottom_layer = [a.index for a in atoms if a.tag == 3]
constraint = FixAtoms(indices=bottom_layer)
atoms.set_constraint(constraint)



surface_indices = [a.index for a in atoms if a.tag == 1]

box_height = 6.0
box = [atoms.cell[0], atoms.cell[1], np.array([0, 0, box_height])]
z_shift = atoms[surface_indices[0]].position[2]- (box_height*0.5)

atoms_in_box = [a for a in atoms if a.position[2] < z_shift+3 and a.position[2] > z_shift-3]


####### PICK PARAMETERS FOR RELAXATION VOLUME AND IMMEDIATE REJECTIONS ####
r_relax = 1.98
r_min, r_max = 0.5,2.9
volume = get_volume(box)
#print(volume)

###### SET THE CALCULATOR WITH MINIMIZATION OR NOT #######
relax=False
path_to_model='/Users/emanuele/MLIP/mace-large-density-agnesi-stress.model'
if relax:
    calculator = MACE_F_Calculator(
                model_paths=path_to_model,
                steps=40,
                fmax=0.2,
                cueq=False,
                device='cpu',
                )
    volume -= 4/3 * np.pi * (r_relax**3) * len(atoms_in_box)
    species_bias = {'Ag': r_relax}
else:
    calculator = MACECalculator(
                model_paths=path_to_model,
                device='cpu'
                )
    species_bias = {}

species = ['Ag', 'O']

mu_ag = -2.8224
delta_mu_o = -0.4
mu_o = -4.925 + delta_mu_o
out_name = 'fcc111_mu'+str(delta_mu_o)
if relax:
    out_name = 'relax_'+out_name

num_deletions = 1
num_insertions = 1
num_thermal = num_deletions + num_insertions

if relax:
    move_list = [[num_deletions, num_insertions],
             [DeletionMove(species=species,
                           seed=1234678764,
                           operating_box=box,
                           z_shift=z_shift),
              InsertionMove(species=species,
                                seed=6758763657,
                                operating_box=box,
                                min_max_insert=[r_min,r_max],
                                z_shift=z_shift,
                                insertion_mode='box')]]
else:
    move_list = [[num_deletions, num_insertions, num_thermal],
             [DeletionMove(species=species,
                           seed=1234678764,
                           operating_box=box,
                           z_shift=z_shift),
              InsertionMove(species=species,
                                seed=6758763657,
                                operating_box=box,
                                min_max_insert=[r_min,r_max],
                                z_shift=z_shift,
                                insertion_mode='box'),
              DisplacementMove(species=species, 
                               seed=45432052, 
                               max_displacement=0.4)]]



move_selector = MoveSelector(*move_list)
T = 500
gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            calculator=calculator,
            mu={'Ag' : mu_ag, 'O' : mu_o},
            units_type='metal',
            species=species,
            species_bias=species_bias,
            volume=volume,
            temperature=T,
            move_selector=move_selector,
            outfile=out_name+'.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=out_name+'.xyz')

gcmc.run(1000000)
