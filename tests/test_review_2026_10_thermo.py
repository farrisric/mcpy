"""
Review 2026-10, thermo group: plot_phase_diagram input checks and
derive_mu_bulk fit. torch/mace-free.

Run with: python -m pytest tests/test_review_2026_10_thermo.py -v
"""
import matplotlib
matplotlib.use('Agg')  # must precede any pyplot import; hence the E402 below

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.eos import birchmurnaghan  # noqa: E402

from mcpy.utils import derive_mu_bulk  # noqa: E402
from mcpy.utils.phase_diagram import plot_phase_diagram  # noqa: E402


def _frame(n_co, energy):
    atoms = Atoms('Cu4', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
                  cell=[12, 12, 12])
    for i in range(n_co):
        atoms += Atoms('CO', positions=[[4 + 2 * i, 4, 4], [4 + 2 * i, 4, 5.1]])
    atoms.info['energy'] = energy
    return atoms


_FRAMES = [_frame(0, 0.0), _frame(1, -1.0), _frame(2, -1.8)]
_KW = dict(adsorbate='C', metal_symbols=('Cu',), mu_ref=-0.5, kind='nano', T=400.0,
           show_structures=False, dmu_standard=-0.6)


@pytest.mark.parametrize('missing', ['adsorbate', 'metal_symbols', 'mu_ref'])
def test_plot_phase_diagram_has_no_lab_defaults(missing):
    # thermo-postproc-5: no silent H / (Pd, Ag) / -3.05 eV defaults.
    kwargs = {k: v for k, v in _KW.items() if k != missing}
    with pytest.raises(TypeError, match=missing):
        plot_phase_diagram(_FRAMES, **kwargs)


def test_plot_phase_diagram_rejects_reversed_dmu_range():
    with pytest.raises(ValueError, match='dmu_range'):
        plot_phase_diagram(_FRAMES, dmu_range=(0.0, -1.0), **_KW)


def test_plot_phase_diagram_nano_needs_metal_atoms():
    kwargs = dict(_KW, metal_symbols=('Pd',))
    with pytest.raises(ValueError, match='metal'):
        plot_phase_diagram(_FRAMES, **kwargs)


def test_plot_phase_diagram_skips_non_finite_energies():
    ref = plot_phase_diagram(_FRAMES, **_KW)
    res = plot_phase_diagram([_frame(1, np.nan), _frame(0, -np.inf)] + _FRAMES, **_KW)
    assert res['stoich'] == ref['stoich']
    np.testing.assert_allclose(res['transitions'], ref['transitions'], rtol=0, atol=1e-12)


class _BirchMurnaghanBulk:
    """Per-atom energy from a Birch-Murnaghan EOS, as a real potential gives."""

    E0, B0, BP, V0 = -2.8, 0.62, 5.5, 4.25 ** 3 / 4

    def get_potential_energy(self, atoms):
        v = atoms.get_volume() / len(atoms)
        return len(atoms) * birchmurnaghan(v, self.E0, self.B0, self.BP, self.V0)


def test_derive_mu_bulk_is_unbiased_off_guess():
    # thermo-postproc-6: the parabola in a was biased by meV/atom when the
    # guessed a is a few percent off.
    mu = derive_mu_bulk(_BirchMurnaghanBulk(), 'Ag', a=4.1592)
    assert mu == pytest.approx(_BirchMurnaghanBulk.E0, abs=2e-4)
