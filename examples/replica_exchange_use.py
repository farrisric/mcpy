from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.ensembles import ReplicaExchange
from mcpy.moves import DeletionMove, InsertionMove, DisplacementMove
from mcpy.moves.move_selector import MoveSelector
from ase.calculators.lj import LennardJones
from ase import Atoms


class Calculator():
    def __init__(self) -> None:
        self.calculator = LennardJones(sigma=3.4, epsilon=0.010086, rc=10.2, smooth=True)

    def get_potential_energy(self, atoms):
        atoms.calc = self.calculator
        return atoms.get_potential_energy()

lj = Calculator()
atoms = Atoms('Ar', cell=[27.2, 27.2, 27.2], pbc=True)
lj.get_potential_energy(atoms)

box = atoms.get_cell()
species = ['Ar']

move_list = [[25, 25, 50],
             [DeletionMove(species=species, seed=1242401, operating_box=box),
              InsertionMove(species=species, seed=5958313, operating_box=box),
              DisplacementMove(species=species, seed=9584824, max_displacement=1.7)]]

move_selector = MoveSelector(*move_list)
T = 87.79


def gcmc_factory(T=87.79, mu={'Ar': 0}, rank=0):
    gcmc = GrandCanonicalEnsemble(
                atoms=atoms,
                calculator=lj,
                mu=mu,
                units_type='metal',
                species=species,
                temperature=T,
                move_selector=move_selector,
                outfile=f'replica_{rank}.out',
                trajectory_write_interval=1,
                outfile_write_interval=1,
                traj_file=f'replica_{rank}.xyz')
    return gcmc


mus = [{'Ar': -8.7*0.010086},
    {'Ar': -8.6*0.010086},
    {'Ar': -8.56*0.010086},
    {'Ar': -8.5*0.010086},
    {'Ar': -8.5*0.010086},
    {'Ar': -8.4*0.010086}]

re_gcmc = ReplicaExchange(
        gcmc_factory,
        mus=mus,
        gcmc_steps=300000,
        exchange_interval=500,
        write_out_interval=100)

re_gcmc.run()
