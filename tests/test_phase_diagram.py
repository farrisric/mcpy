"""
Tests for mcpy.utils.phase_diagram.plot_phase_diagram, focused on the
molecular-adsorbate additions: custom axis label and the reservoir
stoichiometry used by the pressure twin axis. torch/mace-free.

Run with: python -m pytest tests/test_phase_diagram.py -v
"""
import matplotlib
matplotlib.use('Agg')

from ase import Atoms

from mcpy.utils.phase_diagram import plot_phase_diagram


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
