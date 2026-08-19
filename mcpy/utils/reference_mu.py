"""Derive reference chemical potentials from the potential at run time.

Reference energies (E(O2)/2, bulk metal energy per atom) differ between
checkpoints, so hardcoded mu values silently change meaning whenever the
potential is swapped. Deriving them with the run's own calculator keeps
``delta_mu`` the only physical knob.
"""
import logging

import numpy as np
from ase.build import bulk, molecule

logger = logging.getLogger(__name__)


def derive_mu_gas(calculator, species: str = 'O2', box: float = 20.0) -> float:
    """Half the molecular gas energy, e.g. mu_O = E(O2)/2.

    The molecule is placed in a ``box`` A cubic cell; a relaxing calculator
    (BaseCalculator, AlchemiFCalculator, ...) also relaxes its geometry.
    """
    mol = molecule(species)
    mol.set_cell([box, box, box])
    mol.center()
    mol.set_pbc(True)
    mu = float(calculator.get_potential_energy(mol)) / len(mol)
    logger.info('Derived mu(%s/%d) = %.4f eV from the potential',
                species, len(mol), mu)
    return mu


def derive_mu_bulk(calculator, symbol: str = 'Ag',
                   crystalstructure: str = 'fcc', a: float = 4.1592,
                   scan: float = 0.04, n_points: int = 7) -> float:
    """Bulk energy per atom at the potential's own lattice constant.

    Scans ``n_points`` lattice constants in ``a * [1-scan, 1+scan]``,
    fits a parabola and evaluates its minimum, so the reference does not
    inherit the lattice constant of whatever potential ``a`` came from.
    """
    scales = np.linspace(1.0 - scan, 1.0 + scan, n_points)
    energies = []
    for s in scales:
        atoms = bulk(symbol, crystalstructure, a=a * s, cubic=True)
        energies.append(float(calculator.get_potential_energy(atoms)) / len(atoms))
    coeffs = np.polyfit(scales, energies, 2)
    s_min = -coeffs[1] / (2.0 * coeffs[0])
    if coeffs[0] <= 0.0 or not scales[0] <= s_min <= scales[-1]:
        logger.warning(
            'Bulk %s energy minimum fell outside the scanned lattice '
            'constants %.3f-%.3f A (fit minimum at scale %.3f); using the '
            'lowest sampled point. Pass a closer ``a``.',
            symbol, a * scales[0], a * scales[-1], s_min)
        mu = float(np.min(energies))
    else:
        mu = float(np.polyval(coeffs, s_min))
        logger.info('Derived mu(%s) = %.4f eV at a = %.4f A from the potential',
                    symbol, mu, a * s_min)
    return mu
