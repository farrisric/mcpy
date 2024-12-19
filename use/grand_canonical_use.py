from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from ase.calculators.lj import LennardJones
from ase import Atoms


class Calculator():
    def __init__(self) -> None:
        self.calculator = LennardJones()

    def get_potential_energy(self, atoms):
        atoms.calc = self.calculator
        return atoms.get_potential_energy()


lj = Calculator()
atoms = Atoms('Ar', cell=[30, 30, 30])
lj.get_potential_energy(atoms)

T = 500
gcmc = GrandCanonicalEnsemble(
            atoms=atoms,
            calculator=lj,
            mu={'Ar': -0.5},
            masses={'Ar': 1},
            species=['Ar'],
            temperature=T,
            moves=[1, 1],
            max_displacement=0.2,
            outfile=f'replica_T{T}.out',
            trajectory_write_interval=1,
            outfile_write_interval=1,
            traj_file=f'replica_T{T}.xyz',
            min_max_insert=[1.5, 3.0])

gcmc.run(1000000)
