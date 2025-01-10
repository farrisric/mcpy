from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from ase.calculators.lj import LennardJones
from ase import Atoms
from mcpy.moves import DeletionMove, InsertionMove, DisplacementMove
from mcpy.moves.move_selector import MoveSelector


class Calculator():
    def __init__(self) -> None:
        self.calculator = LennardJones()

    def get_potential_energy(self, atoms):
        atoms.calc = self.calculator
        return atoms.get_potential_energy()


lj = Calculator()
atoms = Atoms('Ar', cell=[30, 30, 30])
lj.get_potential_energy(atoms)

box = atoms.get_cell()
species = ['Ar']

move_list = [[25, 25, 50],
             [DeletionMove(species=species, seed=12, operating_box=box),
              InsertionMove(species=species, seed=13, operating_box=box),
              DisplacementMove(species=species, seed=14, max_displacement=1.7)]]

move_selector = MoveSelector(*move_list)
T = 500
gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            calculator=lj,
            mu={'Ar': -8.7*0.010086},
            units_type='metal',
            species=species,
            temperature=87.79,
            move_selector=move_selector,
            outfile=f'replica_T{T}.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=f'replica_T{T}.xyz')

gcmc.run(1000000)
