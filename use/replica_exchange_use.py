from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.ensembles import ReplicaExchange
from ase.calculators.lj import LennardJones
from ase import Atoms


class Calculator():
    def __init__(self) -> None:
        self.calculator = LennardJones(sigma=3.4, epsilon=0.01, rc=10.4, smooth=True)

    def get_potential_energy(self, atoms):
        atoms.calc = self.calculator
        return atoms.get_potential_energy()


lj = Calculator()
atoms = Atoms('Ar', cell=[27.2, 27.2, 27.2], pbc=True)
lj.get_potential_energy(atoms)


def gcmc_factory(T=88, mu={'Ar': 0}, rank=0):
    gcmc = GrandCanonicalEnsemble(
                atoms=atoms,
                calculator=lj,
                mu=mu,
                masses={'Ar': 39.948/6.022e26},
                species=['Ar'],
                temperature=T,
                moves=[50, 50],
                max_displacement=0.2,
                outfile=f'replica_{rank}.out',
                trajectory_write_interval=10,
                outfile_write_interval=10,
                traj_file=f'replica_{rank}.xyz',
                min_max_insert=[0, 27.2])
    return gcmc


temperatures = [200, 250, 300, 350, 400, 450]

mus = [{'Ar': -9/100},
       {'Ar': -8.9/100},
       {'Ar': -8.8/100},
       {'Ar': -8.7/100},
       {'Ar': -8.6/100},
       {'Ar': -8.5/100}]

pt_gcmc = ReplicaExchange(
        gcmc_factory,
        mus=mus,
        gcmc_steps=1000000,
        exchange_interval=5,
        write_out_interval=100)

pt_gcmc.run()
