from ase.build import fcc111, bulk, molecule
from ase.constraints import FixAtoms
import numpy as np

from mcpy.moves import DeletionMove
from mcpy.moves import InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.utils.utils import get_volume, total_volume_with_overlap

R_RELAX = 2.068 

atoms = fcc111('Ag', a=4.165, size=(4, 4, 3), periodic=True, vacuum=8)
bottom_layer = [a.index for a in atoms if a.tag == 3]
constraint = FixAtoms(indices=bottom_layer)
atoms.set_constraint(constraint)

surface_indices = [a.index for a in atoms if a.tag == 1]
box = [atoms.cell[0], atoms.cell[1], np.array([0, 0, 7])]
z_shift = atoms[surface_indices[0]].position[2]-R_RELAX

atoms_in_box = [a for a in atoms if a.position[2] > z_shift]

volume = get_volume(box)
print(volume)
volume_with_overlap = total_volume_with_overlap(
    [R_RELAX for _ in range(len(atoms_in_box))],
    [a.position for a in atoms_in_box],)

volume = volume - volume_with_overlap
print(volume_with_overlap)
print(volume)

calculator = MACE_F_Calculator(
                model_paths='/home/riccardo/Downloads/mace-small-density-agnesi-stress.model',
                steps=20,
                fmax=0.1,
                cueq=False,
                device='cpu',
                )

species = ['Ag', 'O']

move_list = [[1, 1],
             [DeletionMove(species=species,
                           seed=1234678764,
                           operating_box=box,
                           z_shift=z_shift),
              InsertionMove(species=species,
                            seed=6758763657,
                            operating_box=box,
                            min_max_insert=[0.5, 3],
                            z_shift=z_shift)]]

move_selector = MoveSelector(*move_list)

o2 = molecule('O2')
ag = bulk('Ag', a=4.165)
calculator.steps = 100
calculator.fmax = 0.05
e_o2 = calculator.get_potential_energy(o2)
e_ag = calculator.get_potential_energy(ag)

mus = {'Ag': e_ag, 'O': e_o2/2}
delta_mu_O = -0.3
mus['O'] += delta_mu_O
T = 500

calculator.steps = 20
calculator.fmax = 0.1

gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            calculator=calculator,
            mu=mus,
            units_type='metal',
            species=species,
            species_bias={'Ag': R_RELAX},
            volume=volume,
            temperature=T,
            move_selector=move_selector,
            outfile=f'gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_O}.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=f'gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_O}.xyz',
            box=box,
            z_shift=z_shift)

gcmc.run(1000000)