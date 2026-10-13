"""
Tests for mcpy.utils.phase_diagram.plot_phase_diagram, focused on the
molecular-adsorbate additions: custom axis label and the reservoir
stoichiometry used by the pressure twin axis. torch/mace-free.

Run with: python -m pytest tests/test_phase_diagram.py -v
"""
import matplotlib
matplotlib.use('Agg')  # must precede any pyplot import; hence the E402 below

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from ase import Atoms  # noqa: E402

from mcpy.utils.phase_diagram import plot_phase_diagram  # noqa: E402


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
