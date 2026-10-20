"""Padding the MACE input to bucketed atom counts (cuEq per-size tuning).

The pure helpers run wherever nvalchemi imports. The exactness check against
an unpadded forward needs a GPU and a cuEq-capable checkpoint, so it is
opt-in like the other GPU tests (``MCPY_GPU_TESTS=1``).
"""
import contextlib
import os
import pathlib
import sys

import numpy as np
import pytest
from ase import Atoms

# Skip on nvalchemi first: importing torch can SIGBUS in a torch-less env
# (see tests/test_audit_2026_10_calculators.py).
pytest.importorskip('nvalchemi')
alchemi = pytest.importorskip('mcpy.calculators.alchemi')
torch = alchemi.torch


def test_pad_bucket_bounds_the_overhead():
    for n in range(1, 20000, 7):
        m = alchemi._pad_bucket(n)
        assert m >= n
        assert m - n < max(32, m / 8)        # at most ~1/8 extra atoms
    assert alchemi._pad_bucket(1000) == alchemi._pad_bucket(1023)   # one bucket
    assert len({alchemi._pad_bucket(n) for n in range(1025, 2049)}) == 8


def _fake_inputs(n=5, b=2, e=7):
    batch = torch.tensor([0] * 3 + [1] * (n - 3))
    return {
        'positions': torch.randn(n, 3, requires_grad=True),
        'node_attrs': torch.eye(4)[torch.randint(0, 4, (n,))],
        'batch': batch,
        'ptr': torch.tensor([0, 3, n]),
        'edge_index': torch.randint(0, n, (2, e)),
        'shifts': torch.zeros(e, 3),
        'cell': torch.eye(3).repeat(b, 1, 1),
        'head': torch.tensor([1, 1]),
    }


def test_pad_inputs_adds_one_isolated_graph():
    d = _fake_inputs()
    p, n, b = alchemi._pad_inputs(d)
    total = alchemi._pad_bucket(n + 1)
    assert (n, b) == (5, 2)
    assert p['positions'].shape == (total, 3) and p['node_attrs'].shape[0] == total
    assert p['batch'][n:].eq(b).all() and p['batch'].shape[0] == total
    assert p['ptr'].tolist() == [0, 3, 5, total]
    assert p['cell'].shape == (b + 1, 3, 3) and p['head'].tolist() == [1, 1, 1]
    # No edges to the dummies, and the real rows are untouched.
    assert p['edge_index'] is d['edge_index'] and p['shifts'] is d['shifts']
    assert torch.equal(p['positions'][:n], d['positions'])
    # Forces need a gradient path back to the real positions.
    p['positions'].sum().backward()
    assert d['positions'].grad is not None


def test_missing_cueq_helper_disables_padding(monkeypatch):
    # Older/newer nvalchemi without the private helper: fall back to unpadded.
    monkeypatch.setitem(sys.modules, 'nvalchemi.models.mace', None)
    wrapper = object()
    assert alchemi._with_cueq_padding(wrapper) is wrapper


def test_unpad_outputs_keeps_the_real_rows():
    n, b, total = 5, 2, 32
    raw = {'energy': torch.arange(b + 1.0), 'forces': torch.randn(total, 3),
           'node_energy': torch.randn(total), 'stress': torch.randn(b + 1, 3, 3),
           'other': 'kept'}
    out = alchemi._unpad_outputs(raw, n, b)
    assert out['energy'].tolist() == [0.0, 1.0]
    assert out['forces'].shape == (n, 3) and out['node_energy'].shape == (n,)
    assert out['stress'].shape == (b, 3, 3) and out['other'] == 'kept'


# ------------------------------------------------------------- GPU (opt-in)

CHECKPOINT = pathlib.Path(os.environ.get(
    'MCPY_TEST_CHECKPOINT',
    pathlib.Path.home() / '.cache/mace/macesmalldensityagnesistressmodel'))
gpu = pytest.mark.skipif(
    not os.environ.get('MCPY_GPU_TESTS') or not CHECKPOINT.exists()
    or not torch.cuda.is_available(),
    reason='set MCPY_GPU_TESTS=1, provide a MACE checkpoint and a GPU')


def _structures():
    from ase.build import fcc111
    rng = np.random.default_rng(0)
    out = []
    for k in (0, 3, 7):
        at = fcc111('Ag', size=(4, 4, 3), a=4.165, vacuum=6.0)
        at.rattle(0.05, seed=k)
        for _ in range(k):
            at.append('O')
            at.positions[-1] = [rng.random() * 11, rng.random() * 10,
                                at.positions[:, 2].max() + 1.5]
        out.append(at)
    return out


@contextlib.contextmanager
def _unpadded(wrapper):
    """Run ``wrapper`` without padding inside the block, then pad it again."""
    wrapper.__class__ = next(c for c in type(wrapper).__mro__
                             if not issubclass(c, alchemi._PaddedMixin))
    try:
        yield wrapper
    finally:
        alchemi._with_cueq_padding(wrapper)


@pytest.fixture(scope='module')
def padded_calc():
    calc = alchemi.AlchemiFCalculator(str(CHECKPOINT), device='cuda', enable_cueq=True,
                                      compile_model=False)
    assert isinstance(calc.model, alchemi._PaddedMixin)   # cuEq on: padded
    return calc


def _forward(calc, structures):
    batch = calc._batch(structures)
    alchemi._build_nl(batch, calc._nl_hook())
    return calc.model(batch)


@gpu
def test_padding_does_not_change_energies_or_forces(padded_calc):
    padded = _forward(padded_calc, _structures())
    with _unpadded(padded_calc.model):
        plain = _forward(padded_calc, _structures())
    assert padded['energy'].shape == plain['energy'].shape
    assert padded['forces'].shape == plain['forces'].shape
    # float32 noise only: the kernels may differ between the two sizes.
    np.testing.assert_allclose(padded['atomic_energies'].detach().cpu(),
                               plain['atomic_energies'].detach().cpu(), atol=1e-5)
    np.testing.assert_allclose(padded['forces'].detach().cpu(), plain['forces'].detach().cpu(),
                               atol=1e-4)


@gpu
def test_padding_does_not_change_the_stress(padded_calc):
    active = padded_calc.model.model_config.active_outputs
    if 'stress' not in padded_calc.model.model_config.outputs:
        pytest.skip('this checkpoint has no stress output')
    active.add('stress')
    try:
        padded = _forward(padded_calc, _structures())
        with _unpadded(padded_calc.model):
            plain = _forward(padded_calc, _structures())
    finally:
        active.discard('stress')
    assert padded['stress'].shape == plain['stress'].shape == (3, 3, 3)
    np.testing.assert_allclose(padded['stress'].detach().cpu(),
                               plain['stress'].detach().cpu(), atol=1e-5)


@gpu
def test_padded_md_follows_the_unpadded_trajectory(padded_calc):
    # Same seed, so the same velocities and noise: only float32 rounding differs.
    md = dict(temperature=300.0, steps=10, seed=7)
    a = _structures()[1]
    b = a.copy()
    padded_calc.run_md(a, **md)
    with _unpadded(padded_calc.model):
        padded_calc.run_md(b, **md)
    assert len(a) == len(b)
    assert np.abs(a.positions - b.positions).max() < 1e-3


@gpu
def test_shared_wrapper_is_padded_once(padded_calc):
    cls = type(padded_calc.model)
    other = alchemi.AlchemiCalculator(padded_calc.model, device='cuda')
    assert other.model is padded_calc.model
    assert type(other.model) is cls            # not wrapped a second time
    assert alchemi._with_cueq_padding(other.model) is other.model
    assert type(other.model) is cls


@gpu
def test_empty_structures_around_padded_ones(padded_calc):
    slab = _structures()[1]
    ref = padded_calc.get_potential_energies([slab.copy()])[0]
    out = padded_calc.get_potential_energies([Atoms(), slab.copy(), Atoms()])
    assert out[0] == 0.0 and out[2] == 0.0
    assert out[1] == pytest.approx(ref, abs=1e-3)


@gpu
def test_no_padding_without_cueq():
    calc = alchemi.AlchemiCalculator(str(CHECKPOINT), device='cuda', enable_cueq=False,
                                     compile_model=False)
    assert not isinstance(calc.model, alchemi._PaddedMixin)
