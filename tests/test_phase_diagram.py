"""
Tests for mcpy.utils.phase_diagram: plot_phase_diagram (molecular
adsorbates, envelope, metal counts, pressure twin axis),
analyze_phase_diagram_results and from_mu_to_press. torch/mace-free.

Run with: python -m pytest tests/test_phase_diagram.py -v
"""
import matplotlib
matplotlib.use('Agg')  # must precede any pyplot import; hence the E402 below

import logging  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from ase import Atoms  # noqa: E402
from ase.build import fcc111  # noqa: E402
from ase.calculators.singlepoint import SinglePointCalculator  # noqa: E402
from ase.io import write  # noqa: E402

from mcpy.utils.phase_diagram import (  # noqa: E402
    _janaf_dmu0, analyze_phase_diagram_results, from_mu_to_press, plot_phase_diagram)


def _frame(n_co, energy):
    atoms = Atoms('Cu4', positions=[[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]],
                  cell=[12, 12, 12])
    for i in range(n_co):
        atoms += Atoms('CO', positions=[[4 + 2 * i, 4, 4], [4 + 2 * i, 4, 5.1]])
    atoms.info['energy'] = energy
    return atoms


def test_plot_phase_diagram_molecular_adsorbate(tmp_path):
    frames = [_frame(0, 0.0), _frame(1, -1.0), _frame(2, -1.8)]
    out = str(tmp_path / 'pd.png')
    res = plot_phase_diagram(
        frames, adsorbate='C', metal_symbols=('Cu',), mu_ref=-0.5,
        kind='nano', T=400.0, dmu_range=(-1.0, 0.0), show_structures=False,
        outfile=out, adsorbate_label='CO', atoms_per_reservoir_molecule=1)
    assert res['plot_path'] == out
    assert (tmp_path / 'pd.png').exists()
    assert list(res['stoich']) == [0, 1, 2]


def test_plot_phase_diagram_default_diatomic_convention(tmp_path):
    # Backward-compatible default: no label override, reservoir factor 2.
    frames = [_frame(0, 0.0), _frame(1, -1.0)]
    out = str(tmp_path / 'pd2.png')
    res = plot_phase_diagram(
        frames, adsorbate='C', metal_symbols=('Cu',), mu_ref=-0.5,
        kind='nano', T=400.0, dmu_range=(-1.0, 0.0), show_structures=False,
        outfile=out)
    assert (tmp_path / 'pd2.png').exists()
    assert len(res['phase_order']) >= 1


# Four lines on [-1, 0] (nano, 4 metal atoms): free = (E - n * dmu) / 4 * 1000.
# Crossings at -0.60, -0.58 and -0.20 eV: n=1 wins on a narrow 0.02 eV window.
_LINES = {0: 0.0, 1: -0.60, 2: -1.18, 3: -1.38}


def _plot(**kwargs):
    # show_plot=True is a no-op on Agg and leaves the figure open to inspect.
    res = plot_phase_diagram(
        [_frame(n, e) for n, e in _LINES.items()], adsorbate='C', metal_symbols=('Cu',),
        mu_ref=0.0, kind='nano', dmu_range=(-1.0, 0.0), show_plot=True, **kwargs)
    fig = plt.gcf()
    plt.close(fig)
    return res, fig


def test_transitions_are_exact_line_crossings():
    res, _ = _plot(n_bins=50, show_structures=False)
    assert [res['stoich'][i] for i in res['phase_order']] == [0, 1, 2, 3]
    # Exact crossings, independent of the coarse grid; the narrow n=1 phase
    # (0.02 eV < min_phase_width) is kept, not merged into a neighbour.
    np.testing.assert_allclose(res['transitions'], [-0.60, -0.58, -0.20], rtol=0, atol=1e-12)


def test_every_envelope_kink_is_a_transition():
    res, _ = _plot(n_bins=2001, show_structures=False)
    x, y = res['dmu_grid'], res['min_gamma']
    dx = x[1] - x[0]
    kinks = x[1:-1][np.abs(np.diff(y, 2)) > 1e-6]
    assert len(kinks) > 0
    for k in kinks:
        assert np.min(np.abs(res['transitions'] - k)) <= dx
    for t in res['transitions']:
        assert np.min(np.abs(kinks - t)) <= dx


def test_adjacent_phases_get_different_colours():
    res, fig = _plot(show_structures=True)
    ax = fig.axes[0]
    spans = sorted((p for p in ax.patches if not p.get_hatch()), key=lambda p: p.get_x())
    assert len(spans) == len(res['phase_order']) == 4
    colours = [tuple(p.get_facecolor()) for p in spans]
    assert all(a != b for a, b in zip(colours, colours[1:]))
    # Thumbnails carry the same numbers as the regions on the plot.
    titles = [a.get_title() for a in fig.axes if a.get_title()]
    assert [t.split(':')[0] for t in titles] == ['1', '2', '3', '4']


def test_hatched_region_renders(tmp_path):
    out = tmp_path / 'hatch.png'
    _, fig = _plot(show_structures=False, hatch_from=-0.3,
                   hatch_label='bulk oxide stable', outfile=str(out))
    ax = fig.axes[0]
    hatched = [p for p in ax.patches if p.get_hatch()]
    assert len(hatched) == 1 and hatched[0].get_x() == -0.3
    assert 'bulk oxide stable' in [t.get_text() for t in ax.get_legend().get_texts()]
    assert out.stat().st_size > 0


def _ag_slab(n_o, energy, remove_ag=0):
    """Ag48 fcc(111) slab with ``n_o`` O atoms above it, ``remove_ag`` top Ag removed."""
    atoms = fcc111('Ag', size=(4, 4, 3), vacuum=10.0)
    del atoms[[a.index for a in atoms if a.tag == 1][:remove_ag]]
    z = atoms.positions[:, 2].max() + 1.2
    for i in range(n_o):
        atoms += Atoms('O', positions=[[1.5 * i, 1.0, z]])
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    return atoms


_AG_FRAMES = ((0, -135.80, 0), (4, -157.00, 0), (4, -154.40, 1))


def test_metal_count_enters_the_grand_potential():
    # Omega(Ag47O4) - Omega(Ag48O4) = -154.40 - (-2.82) * (47 - 48) + 157.00 = -0.22 eV,
    # so Ag47O4 wins. Its line crosses the clean one at
    # dmu = (-154.40 - 2.82 + 4 * 4.913 + 135.80) / 4 = -0.442 eV (Ag48O4: -0.387 eV).
    # Ag49 has the lowest raw E of the clean frames, but its Omega is higher:
    # -138.50 > -135.80 - 2.82 = -138.62. So Ag48 is the reference.
    ag49 = _ag_slab(0, -138.50)
    ag49 += Atoms('Ag', positions=[[0.0, 0.0, ag49.positions[:, 2].max() + 2.4]])
    ag49.calc = SinglePointCalculator(ag49, energy=-138.50)
    frames = [ag49] + [_ag_slab(*f) for f in _AG_FRAMES]
    res = plot_phase_diagram(frames, adsorbate='O', metal_symbols=('Ag',), mu_ref=-4.913,
                             mu_metal={'Ag': -2.82}, T=500.0, show_structures=False)
    np.testing.assert_allclose(res['transitions'], [-0.442], rtol=0, atol=1e-9)
    assert res['phase_metal_counts'] == [{'Ag': 48}, {'Ag': 47}]
    assert res['min_gamma'][0] == 0.0
    np.testing.assert_allclose(res['phase_ratios'], [0.0, 4 / 47], rtol=0, atol=1e-15)


def test_varying_metal_count_needs_mu_metal():
    frames = [_ag_slab(*f) for f in _AG_FRAMES]
    with pytest.raises(ValueError, match=r'\[47, 48\] Ag atoms.*mu_metal'):
        plot_phase_diagram(frames, adsorbate='O', metal_symbols=('Ag',), mu_ref=-4.913,
                           show_structures=False)
    # A mu_metal key that is not in metal_symbols has no effect, so it is an error.
    with pytest.raises(ValueError, match=r"\['Ag'\].*metal_symbols"):
        plot_phase_diagram(frames, adsorbate='O', metal_symbols=('Pd',), mu_ref=-4.913,
                           mu_metal={'Ag': -2.82}, show_structures=False)


def test_janaf_dmu0_matches_reuter_scheffler_table():
    # Reuter and Scheffler, PRB 65, 035406 (2001), Table I: Delta mu_O(T, 1 atm), 0.01 eV steps.
    table_i = [-0.08, -0.17, -0.27, -0.38, -0.50, -0.61, -0.73, -0.85, -0.98, -1.10]
    ours = [_janaf_dmu0('O2', t, 2) for t in range(100, 1001, 100)]
    np.testing.assert_allclose(ours, table_i, rtol=0, atol=0.0075)
    with pytest.raises(ValueError, match='outside the NIST-JANAF O2 table'):
        _janaf_dmu0('O2', 1500.0, 2)


def _dmu_at_p0(ax2, x0, x1):
    """The dmu of the main axis where the twin axis reads log10(p/p0) = 0."""
    l0, l1 = ax2.get_xlim()
    return x0 - l0 * (x1 - x0) / (l1 - l0)


@pytest.mark.parametrize('T, dmu_standard, dmu0',
                         [(300.0, None, -0.27), (500.0, None, -0.50), (500.0, -0.8, -0.8)])
def test_pressure_axis_is_p0_at_standard_dmu(T, dmu_standard, dmu0):
    # p = p0 at dmu = Delta mu_O^0(T), not at dmu = 0 (10 decades at 500 K).
    # A given dmu_standard wins over the built-in table.
    plot_phase_diagram([_ag_slab(0, -135.80), _ag_slab(4, -157.00)], adsorbate='O',
                       metal_symbols=('Ag',), mu_ref=-4.913, T=T, show_structures=False,
                       show_plot=True, dmu_standard=dmu_standard)
    fig = plt.gcf()
    plt.close(fig)
    assert abs(_dmu_at_p0(fig.axes[1], -1.0, 0.0) - dmu0) < 0.0075


def test_pressure_axis_outside_janaf_range(caplog):
    # The phases do not depend on T: outside the table only the axis goes.
    with caplog.at_level(logging.WARNING, logger='mcpy.utils.phase_diagram'):
        res = plot_phase_diagram([_ag_slab(0, -135.80), _ag_slab(4, -157.00)], adsorbate='O',
                                 metal_symbols=('Ag',), mu_ref=-4.913, T=1200.0,
                                 show_structures=False, show_plot=True)
    fig = plt.gcf()
    plt.close(fig)
    assert len(fig.axes) == 1
    assert 'T = 1200 K' in caplog.text
    np.testing.assert_allclose(res['transitions'], [-0.387], rtol=0, atol=1e-9)


def test_pressure_axis_needs_a_standard_dmu(caplog):
    frames = [_frame(0, 0.0), _frame(1, -1.0)]
    kwargs = dict(adsorbate='C', adsorbate_label='CO', atoms_per_reservoir_molecule=1,
                  metal_symbols=('Cu',), mu_ref=-0.5, kind='nano', T=400.0,
                  show_structures=False, show_plot=True)
    with caplog.at_level(logging.WARNING, logger='mcpy.utils.phase_diagram'):
        plot_phase_diagram(frames, **kwargs)
    fig = plt.gcf()
    plt.close(fig)
    assert len(fig.axes) == 1
    assert 'dmu_standard' in caplog.text
    plot_phase_diagram(frames, dmu_standard=-0.8, **kwargs)
    fig = plt.gcf()
    plt.close(fig)
    assert abs(_dmu_at_p0(fig.axes[1], -1.0, 0.0) - (-0.8)) < 1e-9


@pytest.mark.parametrize('T, dmu0', [(300.0, -0.27), (500.0, -0.50)])
def test_from_mu_to_press_follows_temperature(T, dmu0):
    # At mu = E(O2)/2 + Delta mu_O^0(T) the O2 pressure is p0 at every T.
    u_dft = -9.8612 / 2
    tol = 2 * 0.0075 / (8.617e-5 * T * np.log(10.0))
    assert abs(np.log10(from_mu_to_press(u_dft + dmu0, T=T))) < tol


def test_analyze_bulk_oxide_boundary_and_area(tmp_path):
    # Default references: E(O2)/2 = E(Ag2O) - 2 E(Ag) + h_f, independent of mu_Ag.
    e_o2_half = -3.6312459783333333 * 3 - 2 * -2.82894684 + 0.323
    clean = fcc111('Ag', size=(2, 2, 3), vacuum=10.0)
    clean.info.clear()  # adsorbate_info does not survive extxyz
    clean.calc = SinglePointCalculator(clean, energy=-30.0)
    z_o = clean.positions[:, 2].max() + 1.2
    oxo = clean.copy() + Atoms('O', positions=[[0.0, 0.0, z_o]])
    oxo.calc = SinglePointCalculator(oxo, energy=-30.0 + e_o2_half - 0.6)
    oxo2 = oxo.copy() + Atoms('O', positions=[[1.5, 0.0, z_o]])
    oxo2.calc = SinglePointCalculator(oxo2, energy=-30.0 + 2 * e_o2_half - 0.8)
    path = tmp_path / 'frames.xyz'
    write(path, [clean, oxo, oxo2])
    delta = -0.1
    res = analyze_phase_diagram_results(
        trajectory_path=str(path), idx_ref=0, delta_mu_host=delta, n_bins=1001, T=300,
        save_plot=False, show_plot=True)
    fig = plt.gcf()
    plt.close(fig)
    # The O lines cross at -0.6 and -0.2 eV for any delta_mu_host.
    np.testing.assert_allclose(res['transitions_delta_mu_o'], [-0.6, -0.2], rtol=0, atol=1.001e-3)
    # Bulk Ag2O is stable where 2 (E_Ag + delta) + E(O2)/2 + dmu >= E(Ag2O).
    hatch = [p for p in fig.axes[0].patches if p.get_hatch()]
    assert hatch[0].get_x() == pytest.approx(-0.323 - 2 * delta)
    # Both transitions are below that boundary (-0.123 eV), so both get a line.
    vlines = [ln.get_xdata()[0] for ln in fig.axes[0].lines
              if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]]
    np.testing.assert_allclose(vlines, res['transitions_delta_mu_o'], rtol=0, atol=0)
    # gamma is per |a x b|, the true area of the hexagonal cell.
    c = clean.get_cell()
    area = np.linalg.norm(np.cross(c[0], c[1]))
    assert res['min_energy_mev_per_a2'][-1] == pytest.approx(-800.0 / area)
    assert abs(_dmu_at_p0(fig.axes[1], -1.0, -0.2) - (-0.27)) < 0.0075

    # delta < -h_f / 2 puts the boundary above 0, so there is no hatch. Outside
    # the NIST-JANAF range, only the pressure axis goes.
    res = analyze_phase_diagram_results(
        trajectory_path=str(path), idx_ref=0, delta_mu_host=-0.2, T=1200,
        save_plot=False, show_plot=True)
    fig = plt.gcf()
    plt.close(fig)
    assert not [p for p in fig.axes[0].patches if p.get_hatch()]
    assert len(fig.axes) == 1
    assert len(res['transitions_delta_mu_o']) == 2
