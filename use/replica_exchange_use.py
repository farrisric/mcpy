from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.ensembles import ReplicaExchange
from ase.calculators.lj import LennardJones
from ase import Atoms


class Calculator():
    def __init__(self) -> None:
        self.calculator = LennardJones(smooth=True)

    def get_potential_energy(self, atoms):
        atoms.calc = self.calculator
        return atoms.get_potential_energy()


lj = Calculator()
atoms = Atoms('Ar', cell=[15, 15, 15], pbc=True)
lj.get_potential_energy(atoms)


def gcmc_factory(T):
    gcmc = GrandCanonicalEnsemble(
                atoms=atoms,
                calculator=lj,
                mu={'Ar': 0},
                masses={'Ar': 1},
                species=['Ar'],
                temperature=T,
                moves=[10, 10],
                max_displacement=0.2,
                outfile=f'replica_T{T}.out',
                trajectory_write_interval=10,
                outfile_write_interval=10,
                traj_file=f'replica_T{T}.xyz',
                min_max_insert=[1.5, 3.0])
    return gcmc


temperatures = [300, 400]

pt_gcmc = ReplicaExchange(
        gcmc_factory,
        temperatures,
        gcmc_steps=1000000,
        exchange_interval=50)

pt_gcmc.run()
