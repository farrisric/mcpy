"""Opt-in periodic GPU cache flush (mcpy.utils.gpu + ensemble wiring).

Covers the cadence predicate, the torch-free safety of the flush helper, and
that GrandCanonicalEnsemble._run fires the flush only on the configured
interval. Torch is never imported, so this stays in the lightweight CI matrix.

Run with: python -m pytest tests/test_empty_cache.py -v
"""
import types

import mcpy.ensembles.grand_canonical_ensemble as gce
from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
from mcpy.utils.gpu import empty_cuda_cache, should_empty_cache


def test_should_empty_cache_cadence():
    assert should_empty_cache(5, 5)
    assert should_empty_cache(10, 5)
    assert not should_empty_cache(0, 5)     # never on step 0
    assert not should_empty_cache(3, 5)     # not due yet


def test_should_empty_cache_disabled():
    assert not should_empty_cache(5, 0)     # 0 = off
    assert not should_empty_cache(5, -1)    # negative = off


def test_empty_cuda_cache_safe_without_torch():
    # Never raises; returns False when torch/CUDA is unavailable (CI case).
    assert empty_cuda_cache() in (True, False)


def _gcmc_runnable(interval):
    """A GrandCanonicalEnsemble carrying only what ``_run`` touches."""
    e = GrandCanonicalEnsemble.__new__(GrandCanonicalEnsemble)
    e._step = 0
    e._last_step_seconds = 0.0
    e._outfile_write_interval = 10 ** 9      # suppress writes during the test
    e._trajectory_write_interval = 10 ** 9
    e._empty_cache_interval = interval
    e.n_atoms = 0
    e.E_old = 0.0
    e.logger = types.SimpleNamespace(info=lambda *a, **k: None)
    e.do_gcmc_step = lambda: None
    e.write_outfile = lambda: None
    e.write_coordinates = lambda *a, **k: None
    return e


def test_run_flushes_on_interval(monkeypatch):
    calls = []
    monkeypatch.setattr(gce, 'empty_cuda_cache', lambda: calls.append(1))
    e = _gcmc_runnable(interval=3)
    for _ in range(9):
        e._run()
    assert len(calls) == 3          # fired at steps 3, 6, 9


def test_run_no_flush_when_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(gce, 'empty_cuda_cache', lambda: calls.append(1))
    e = _gcmc_runnable(interval=0)
    for _ in range(9):
        e._run()
    assert calls == []
