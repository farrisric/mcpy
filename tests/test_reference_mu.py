"""
Tests for runtime reference-mu derivation. Uses a stub calculator so the
suite stays free of torch/mace, matching the other tests.

Run with: python -m pytest tests/test_reference_mu.py -v
"""
import numpy as np
import pytest

from mcpy.utils import derive_mu_bulk, derive_mu_gas


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
