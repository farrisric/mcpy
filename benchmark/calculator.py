from mcpy.calculators import MACECalculator
from ase.cluster import Octahedron
import time
import numpy as np

calculator = MACECalculator('/home/riccardo/Downloads/mace-large-density-agnesi-stress.model')

n = 12
times = []

for _ in range(n):
    atoms = Octahedron('Ag', 7, 1)
    start = time.time()
    calculator.get_potential_energy(atoms)
    end = time.time()
    times.append(end-start)

print(f'Average time for MACE calculation: {sum(times[2:])/n} s')
print(f'Standard deviation: {np.std(times)} s')
