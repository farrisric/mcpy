"""Derive reference chemical potentials from the potential at run time.

Reference energies (E(O2)/2, bulk metal energy per atom) differ between
checkpoints, so hardcoded mu values silently change meaning whenever the
potential is swapped. Deriving them with the run's own calculator keeps
``delta_mu`` the only physical knob.
"""
import logging

import numpy as np
from ase.build import bulk, molecule
from ase.eos import EquationOfState
from ase.units import kB

logger = logging.getLogger(__name__)


def derive_mu_gas(calculator, species: str = 'O2', box: float = 20.0) -> float:
    """Per-atom molecular gas energy, e.g. mu_O = E(O2)/2.

    Only defined for homonuclear molecules (O2, H2, N2, ...): for a
    heteronuclear one such as CO, E/len(mol) is no single element's
    chemical potential, so a ``ValueError`` is raised instead.

    The molecule is placed in a ``box`` A cubic cell; a relaxing calculator
    (BaseCalculator, AlchemiFCalculator, ...) also relaxes its geometry.
    """
    mol = molecule(species)
    if len(set(mol.get_chemical_symbols())) > 1:
        raise ValueError(
            f'derive_mu_gas needs a homonuclear molecule, got {species!r}: '
            'E/len(mol) of a heteronuclear molecule is not the chemical '
            'potential of any one element. Derive it from a reference '
            'reaction instead.')
    mu = _molecule_energy(calculator, mol, box) / len(mol)
    logger.info('Derived mu(%s/%d) = %.4f eV from the potential',
                species, len(mol), mu)
    return mu


def _molecule_energy(calculator, mol, box):
    mol.set_cell([box, box, box])
    mol.center()
    mol.set_pbc(True)
    return float(calculator.get_potential_energy(mol))


def mu_h_che(calculator, ph: float, u_she: float = 0.0,
             temperature: float = 298.15, box: float = 20.0) -> float:
    """mu of (H+ + e-) from the computational hydrogen electrode (CHE).

    mu_H = E(H2)/2 - e*U_SHE - kT*ln(10)*pH

    pH and potential only enter as U_RHE = U_SHE + kT*ln(10)*pH / e, so a
    fixed U_RHE makes pH drop out. Bare energies: no ZPE or entropy terms.
    """
    return (derive_mu_gas(calculator, 'H2', box)
            - u_she - kB * temperature * np.log(10) * ph)


def mu_o_che(calculator, ph: float, u_she: float = 0.0,
             temperature: float = 298.15, box: float = 20.0) -> float:
    """mu of O exchanged with liquid water, via the CHE.

    H2O -> O* + 2(H+ + e-), so mu_O = E(H2O) - 2*mu_H. Insert O with this
    and H with :func:`mu_h_che` at the same pH and U_SHE to get O/OH/H2O
    coverages. Bare energies: no ZPE or entropy terms.
    """
    e_h2o = _molecule_energy(calculator, molecule('H2O'), box)
    mu = e_h2o - 2.0 * mu_h_che(calculator, ph, u_she, temperature, box)
    logger.info('CHE mu(O) = %.4f eV at pH %.2f, U_SHE %.3f V', mu, ph, u_she)
    return mu


def derive_mu_bulk(calculator, symbol: str = 'Ag',
                   crystalstructure: str = 'fcc', a: float = 4.1592,
                   scan: float = 0.04, n_points: int = 7) -> float:
    """Bulk energy per atom at the potential's own lattice constant.

    Scans ``n_points`` lattice constants in ``a * [1-scan, 1+scan]`` and
    fits an equation of state (``ase.eos``, SJEOS) to E(V), so the reference
    does not inherit the lattice constant of whatever potential ``a`` came
    from. The cell is the cubic conventional cell, so ``crystalstructure``
    must be cubic (fcc, bcc, sc, diamond, ...); hcp is not supported.
    """
    scales = np.linspace(1.0 - scan, 1.0 + scan, n_points)
    volumes, energies = [], []
    for s in scales:
        atoms = bulk(symbol, crystalstructure, a=a * s, cubic=True)
        volumes.append(atoms.get_volume() / len(atoms))
        energies.append(float(calculator.get_potential_energy(atoms)) / len(atoms))
    try:
        v0, mu, _ = EquationOfState(volumes, energies).fit()
    except ValueError:  # SJEOS: no minimum
        v0 = np.nan
    if not min(volumes) <= v0 <= max(volumes):
        logger.warning(
            'Bulk %s energy minimum fell outside the scanned lattice '
            'constants %.3f-%.3f A; using the lowest sampled point. '
            'Pass a closer ``a``.', symbol, a * scales[0], a * scales[-1])
        mu = float(np.min(energies))
    else:
        mu = float(mu)
        logger.info('Derived mu(%s) = %.4f eV at a = %.4f A from the potential',
                    symbol, mu, a * (v0 / volumes[0]) ** (1 / 3) * scales[0])
    return mu
