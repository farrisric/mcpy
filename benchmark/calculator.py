"""Time MACECalculator energy evaluations on an Ag octahedron.

    python benchmark/calculator.py --model /path/to/mace.model
"""
import argparse
import time

import numpy as np
from ase.cluster import Octahedron

from mcpy.calculators import MACECalculator


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, help='Path to a MACE .model file')
    p.add_argument('--edge', type=int, default=7, help='Octahedron edge length')
    p.add_argument('--n', type=int, default=12, help='Number of timed evaluations')
    args = p.parse_args()

    calculator = MACECalculator(args.model)
    times = []
    for _ in range(args.n):
        atoms = Octahedron('Ag', args.edge, 1)
        start = time.time()
        calculator.get_potential_energy(atoms)
        times.append(time.time() - start)

    print(f'Average time for MACE calculation: {sum(times[2:]) / args.n} s')
    print(f'Standard deviation: {np.std(times)} s')


if __name__ == '__main__':
    main()
