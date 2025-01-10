from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.ensembles import ReplicaExchange
from mcpy.moves import DeletionMove, InsertionMove, DisplacementMove
from mcpy.moves.move_selector import MoveSelector
from mcpy.calculators import MACECalculator

from ase.build import fcc111
import numpy as np

atoms = fcc111('Ag', size=(4, 4, 3), periodic=True, vacuum=8)
surface_indices = [a.index for a in atoms if a.tag == 1]
box = [atoms.cell[0], atoms.cell[1], np.array([0, 0, 6])]
z_shift = atoms[surface_indices[0]].position[2]-3


calculator = MACECalculator('/home/riccardo/Downloads/mace-large-density-agnesi-stress.model')
calculator.get_potential_energy(atoms)

species = ['Ag', 'O']

move_list = [[25, 25, 50],
             [DeletionMove(species=species,
                           seed=12,
                           operating_box=box,
                           z_shift=z_shift),
              InsertionMove(species=species,
                            seed=13,
                            operating_box=box,
                            min_max_insert=[1.5, 3.5],
                            z_shift=z_shift),
              DisplacementMove(species=species,
                               seed=14,
                               max_displacement=0.2)]]

move_selector = MoveSelector(*move_list)


def gcmc_factory(mu, rank=0):
    gcmc = GrandCanonicalEnsemble(
                atoms=atoms,
                calculator=calculator,
                mu=mu,
                units_type='metal',
                species=species,
                temperature=500,
                move_selector=move_selector,
                outfile=f'replica_{rank}.out',
                trajectory_write_interval=1,
                outfile_write_interval=1,
                traj_file=f'replica_{rank}.xyz')
    return gcmc

mus = [{'Ag' : -2.8, 'O' : -4.9},
       {'Ag' : -2.8, 'O' : -5.0},
       {'Ag' : -2.8, 'O' : -5.1},
       {'Ag' : -2.8, 'O' : -5.2}]

pt_gcmc = ReplicaExchange(
        gcmc_factory,
        mus=mus,
        gcmc_steps=300000,
        exchange_interval=500,
        write_out_interval=100)

pt_gcmc.run()
