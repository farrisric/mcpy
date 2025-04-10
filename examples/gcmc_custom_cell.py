from ase.build import fcc111, bulk, molecule
from ase.constraints import FixAtoms

from mcpy.moves import DeletionMove
from mcpy.moves import InsertionMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import CustomCell


atoms = fcc111('Ag', a=4.165, size=(4, 4, 3), periodic=True, vacuum=8)
bottom_layer = [a.index for a in atoms if a.tag == 3]
constraint = FixAtoms(indices=bottom_layer)
atoms.set_constraint(constraint)

cell_ag_ag = CustomCell(atoms, custom_height=7, bottom_z=12.8-2.068,
                        species_radii={'Ag': 2.947, 'O' : 0})

cell_ag_o = CustomCell(atoms, custom_height=7, bottom_z=12.8-2.068,
                       species_radii={'Ag': 2.068, 'O' : 0})


calculator = MACE_F_Calculator(
                model_paths='/home/riccardo/Downloads/mace-small-density-agnesi-stress.model',
                steps=20,
                fmax=0.1,
                cueq=False,
                device='cpu',
                )

species = ['Ag', 'O']

move_list = [[25, 25, 25, 25],
             [DeletionMove(cell_ag_ag,
                           species=['Ag'],
                           seed=12346783764),
              DeletionMove(cell_ag_o,
                           species=['O'],
                           seed=43215423143),
              InsertionMove(cell_ag_ag,
                            species=['Ag'],
                            species_bias=['Ag'],
                            min_insert=0.5,
                            seed=6758763657),
              InsertionMove(cell_ag_o,
                            species=['O'],
                            species_bias=['Ag'],
                            min_insert=0.5,
                            seed=3675437856)]]

move_selector = MoveSelector(*move_list)

o2 = molecule('O2')
ag = bulk('Ag', a=4.165)
calculator.steps = 100
calculator.fmax = 0.05
e_o2 = calculator.get_potential_energy(o2)
e_ag = calculator.get_potential_energy(ag)

mus = {'Ag': e_ag-0.176, 'O': e_o2/2}
delta_mu_O = -0.5
mus['O'] += delta_mu_O
T = 500

calculator.steps = 20
calculator.fmax = 0.1

gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[cell_ag_ag, cell_ag_o],
            calculator=calculator,
            mu=mus,
            units_type='metal',
            species=species,
            temperature=T,
            move_selector=move_selector,
            outfile=f'gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_O}.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=f'gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_O}.xyz')

gcmc.run(1000000)
