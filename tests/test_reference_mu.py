"""
Tests for runtime reference-mu derivation. Uses a stub calculator so the
suite stays free of torch/mace, matching the other tests.

Run with: python -m pytest tests/test_reference_mu.py -v
"""
import numpy as np
import pytest

from mcpy.utils import derive_mu_bulk, derive_mu_gas, mu_h_che, mu_o_che


class StubGasCalculator:
    def get_potential_energy(self, atoms):
        assert atoms.get_chemical_formula() == 'O2'
        assert atoms.pbc.all()
        return -9.8


class StubBulkCalculator:
    """Per-atom energy parabolic in the lattice constant, minimum at a_min."""

    def __init__(self, a_min, e_min):
        self.a_min = a_min
        self.e_min = e_min

    def get_potential_energy(self, atoms):
        a = atoms.cell.lengths()[0]  # cubic=True conventional cell
        return len(atoms) * (self.e_min + 0.5 * (a - self.a_min) ** 2)


def test_derive_mu_gas_is_half_molecule_energy():
    assert derive_mu_gas(StubGasCalculator()) == pytest.approx(-4.9)


def test_derive_mu_bulk_finds_off_guess_minimum():
    # Minimum deliberately off the guess but inside the +-4% scan window.
    calc = StubBulkCalculator(a_min=4.25, e_min=-2.8)
    assert derive_mu_bulk(calc, 'Ag', a=4.1592) == pytest.approx(-2.8, abs=1e-4)


def test_derive_mu_bulk_warns_outside_scan(caplog):
    # Minimum far outside the scan window: fall back to lowest sample.
    calc = StubBulkCalculator(a_min=5.5, e_min=-2.8)
    with caplog.at_level('WARNING'):
        mu = derive_mu_bulk(calc, 'Ag', a=4.1592)
    scan_max = 4.1592 * 1.04
    assert mu == pytest.approx(-2.8 + 0.5 * (scan_max - 5.5) ** 2, abs=1e-4)
    assert any('outside the scanned' in r.message for r in caplog.records)
    assert np.isfinite(mu)


class StubMoleculeCalculator:
    ENERGIES = {'H2': -6.8, 'H2O': -14.2}

    def get_potential_energy(self, atoms):
        return self.ENERGIES[atoms.get_chemical_formula()]


def test_che_h_and_o_shift_with_ph_and_potential():
    calc = StubMoleculeCalculator()
    slope = 0.059159  # kT ln10 at 298.15 K, eV per pH unit
    assert mu_h_che(calc, ph=0) == pytest.approx(-3.4)
    assert mu_h_che(calc, ph=7, u_she=0.5) == pytest.approx(-3.4 - 0.5 - 7 * slope, abs=1e-5)
    # mu_O = E(H2O) - E(H2) + 2U + 2 kT ln10 pH
    assert mu_o_che(calc, ph=7, u_she=0.5) == pytest.approx(
        -14.2 + 6.8 + 1.0 + 14 * slope, abs=1e-5)
    # Fixed U_RHE: pH drops out.
    assert mu_o_che(calc, ph=14, u_she=-14 * slope) == pytest.approx(
        mu_o_che(calc, ph=0), abs=1e-5)
