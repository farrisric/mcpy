from ase.cluster import Octahedron
from ase.build import molecule

from mcpy.moves import DeletionMove, InsertionMove, PermutationMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.calculators import MACE_F_Calculator
from mcpy.cell import SphericalCell


atoms = Octahedron('Pt', 4, 1)
symbols = ['Pt' for _ in range(int(len(atoms)/2))] + ['Au' for _ in range(int(len(atoms)/2))]
atoms.symbols = symbols

scell = SphericalCell(atoms, vacuum=3, species_radii={'Pt': 2, 'Au': 2.5, 'H' : 0},
                      mc_sample_points=100_000)

calculator = MACE_F_Calculator(
                model_paths='/home/riccardo/Downloads/mace-small-density-agnesi-stress.model',
                steps=20,
                fmax=0.1,
                cueq=False,
                device='cpu',
                )

species = ['H']

move_list = [[1, 1, 1],
             [DeletionMove(scell,
                           species=species,
                           seed=43215423143),
              InsertionMove(scell,
                            species=species,
                            min_insert=0.5,
                            seed=3675437856),
              PermutationMove(species=['Au', 'Pt'],
                              seed=32432187412)]]

move_selector = MoveSelector(*move_list)

h2 = molecule('H2')
calculator.steps = 100
calculator.fmax = 0.05
e_h2 = calculator.get_potential_energy(h2)

mus = {'H': e_h2/2}
delta_mu_H = -0.5
mus['H'] += delta_mu_H
T = 500

gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            cells=[scell],
            calculator=calculator,
            mu=mus,
            units_type='metal',
            species=species,
            temperature=T,
            move_selector=move_selector,
            outfile=f'gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_H}.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=f'gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_H}.xyz')

gcmc.run(1000000)
