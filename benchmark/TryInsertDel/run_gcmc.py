from mcpy.moves import DeletionMove, InsertionMove, DisplacementMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mace.calculators import MACECalculator
from ase.optimize import LBFGS
from ase.cluster import Octahedron
from ase.constraints import FixAtoms
from ase.build import fcc111, molecule, bulk
from ase import Atoms
import numpy as np
import sys
from ase.io import read


class Calculator():
    def __init__(self) -> None:
        self.calculator = MACECalculator(
            model_paths='/Users/emanuele/GCMC/CodeBenchmarks/MACETest/mace-large-density-agnesi-stress.model',
            device='cpu')

    def get_potential_energy(self, atoms):
        return self.relax(atoms)

    def relax(self, atoms):
        atoms.calc = self.calculator
        #opt = LBFGS(atoms, logfile=None)
        #opt.run(steps=20, fmax=0.2)
        return atoms.get_potential_energy()


atoms = Octahedron('Ag', 4, 1)
atoms.center(3)
box = atoms.get_cell()
volume_acc = box.volume #- 4/3 * np.pi * (1.5**3) * len(atoms)
calculator = Calculator()
calculator.relax(atoms)

O2 = molecule('O2')
E2 = calculator.relax(O2)

Ag = bulk('Ag', a=4.165)
EAg = calculator.get_potential_energy(Ag)

conv = 1.66053906660e-27

delta_mu = -0.5
mu_oxygen = E2/2 + delta_mu

mu_silver = EAg

species = ['O']
str_species = ''.join(species)
z_shift = 0

move_list = [[5, 5, 10],
             [DeletionMove(species=species,
                           seed=12340241,
                           operating_box=box,
                           z_shift=z_shift),
              InsertionMove(species=species,
                            seed=94948189,
                            operating_box=box,
                            min_max_insert=[1.5,1.3],
                            z_shift=z_shift),
              DisplacementMove(species=['Ag','O'],
                               seed=14,
                               max_displacement=0.15)]]

move_selector = MoveSelector(*move_list)
T = 500
gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            calculator=calculator,
            mu={'O': mu_oxygen},
            units_type='metal',
            species=species,
            temperature=T,
            volume=volume_acc,
            move_selector=move_selector,
            outfile='Hybrid_GCMC.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file='Hybrid_GCMC.xyz')

gcmc.run(300000)
