"""
Tests for mcpy's logging behavior.

Verifies that importing the library has no side effects on the host
application's logging, that the opt-in ``configure`` helper attaches and
removes handlers cleanly, and that library modules log under predictable
names. Also enforces (statically) that no library module calls
``logging.basicConfig`` at import time.

Run with: python -m pytest tests/test_logging.py -v
"""
from __future__ import annotations

import logging
import pathlib
import re
import subprocess
import sys

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LIB_ROOT = REPO_ROOT / "mcpy"


@pytest.fixture
def clean_mcpy_logger():
    """Snapshot and restore the mcpy logger so each test starts clean."""
    log = logging.getLogger("mcpy")
    saved_handlers = list(log.handlers)
    saved_level = log.level
    saved_propagate = log.propagate
    try:
        yield log
    finally:
        log.handlers = saved_handlers
        log.setLevel(saved_level)
        log.propagate = saved_propagate


# ---------------------------------------------------------------------------
# Import side effects
# ---------------------------------------------------------------------------

def test_import_does_not_touch_root_logger():
    """Importing mcpy in a fresh interpreter must not configure root."""
    code = (
        "import logging, sys\n"
        "root = logging.getLogger()\n"
        "before = (list(root.handlers), root.level)\n"
        "import mcpy  # noqa: F401\n"
        "after = (list(root.handlers), root.level)\n"
        "assert before == after, (before, after)\n"
        "print('OK')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK"


def test_mcpy_logger_has_null_handler():
    import mcpy  # noqa: F401
    log = logging.getLogger("mcpy")
    assert any(isinstance(h, logging.NullHandler) for h in log.handlers), (
        f"NullHandler missing from mcpy logger; handlers={log.handlers}"
    )


def test_library_loggers_use_module_name():
    """Each module exposes ``logger = logging.getLogger(__name__)``."""
    from mcpy.ensembles import base_ensemble, canonical_ensemble
    from mcpy.ensembles import grand_canonical_ensemble

    assert base_ensemble.logger.name == "mcpy.ensembles.base_ensemble"
    assert canonical_ensemble.logger.name == "mcpy.ensembles.canonical_ensemble"
    assert grand_canonical_ensemble.logger.name == (
        "mcpy.ensembles.grand_canonical_ensemble"
    )


# ---------------------------------------------------------------------------
# Static checks on library source
# ---------------------------------------------------------------------------

def _lib_modules():
    """All .py files inside mcpy/ that ship as library code."""
    yield from LIB_ROOT.rglob("*.py")


def test_no_basicconfig_in_library():
    """Library modules must never call ``logging.basicConfig`` at import."""
    pattern = re.compile(r"^[^#\n]*logging\.basicConfig", re.MULTILINE)
    offenders = []
    for path in _lib_modules():
        text = path.read_text()
        if pattern.search(text):
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], (
        f"basicConfig found in library modules: {offenders}"
    )


def test_compute_radii_basicconfig_under_main_guard():
    """The calibration script's basicConfig must live under ``if __name__``."""
    path = REPO_ROOT / "scripts" / "compute_radii.py"
    text = path.read_text()
    # Find the offset of the main guard and any basicConfig call.
    guard_idx = text.find("if __name__ == '__main__':")
    basic_idx = text.find("logging.basicConfig")
    assert guard_idx != -1, "missing __main__ guard"
    assert basic_idx != -1, "expected basicConfig inside main guard"
    assert basic_idx > guard_idx, (
        "basicConfig must appear after the if __name__ guard"
    )


# ---------------------------------------------------------------------------
# configure() helper
# ---------------------------------------------------------------------------

def test_configure_attaches_stream_handler(clean_mcpy_logger):
    from mcpy.utils.logging import configure

    log = configure(level=logging.DEBUG)
    stream_handlers = [
        h for h in log.handlers if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert len(stream_handlers) == 1
    assert log.level == logging.DEBUG
    assert log.propagate is False


def test_configure_is_idempotent(clean_mcpy_logger):
    """Calling configure twice must not stack duplicate handlers."""
    from mcpy.utils.logging import configure

    configure(level=logging.INFO)
    first = [h for h in clean_mcpy_logger.handlers
             if getattr(h, "_mcpy_managed", False)]
    configure(level=logging.DEBUG)
    second = [h for h in clean_mcpy_logger.handlers
              if getattr(h, "_mcpy_managed", False)]
    assert len(first) == len(second) == 1
    # Old handler is replaced, not appended.
    assert first[0] is not second[0]


def test_configure_writes_to_file(tmp_path, clean_mcpy_logger):
    from mcpy.utils.logging import configure

    log_path = tmp_path / "mcpy.log"
    log = configure(level=logging.INFO, file=str(log_path), stream=False)
    log.info("hello from test")

    # Close the file handler to flush before reading.
    for h in log.handlers:
        if isinstance(h, logging.FileHandler):
            h.close()

    contents = log_path.read_text()
    assert "hello from test" in contents
    assert "mcpy" in contents  # logger name appears via default formatter


def test_configure_rank_format(clean_mcpy_logger, capsys):
    from mcpy.utils.logging import configure

    log = configure(level=logging.INFO, mpi_rank=3)
    log.info("ranked message")
    captured = capsys.readouterr()
    assert "[rank 3]" in captured.err or "[rank 3]" in captured.out


# ---------------------------------------------------------------------------
# Library logger emission
# ---------------------------------------------------------------------------

def test_module_log_record_propagates_to_mcpy(caplog):
    """A log call inside a library module is captured under the mcpy hierarchy."""
    from mcpy.ensembles import base_ensemble

    with caplog.at_level(logging.ERROR, logger="mcpy"):
        base_ensemble.logger.error("boom")
    matching = [r for r in caplog.records
                if r.name == "mcpy.ensembles.base_ensemble"
                and r.getMessage() == "boom"]
    assert matching, f"expected log record not captured: {caplog.records}"


def test_logger_exception_includes_traceback(caplog):
    """``logger.exception`` should attach exc_info to the record."""
    from mcpy.ensembles import base_ensemble

    with caplog.at_level(logging.ERROR, logger="mcpy"):
        try:
            raise RuntimeError("oops")
        except RuntimeError:
            base_ensemble.logger.exception("Error writing to file %s", "out.txt")

    rec = next(r for r in caplog.records if "Error writing" in r.getMessage())
    assert rec.exc_info is not None
    assert rec.exc_info[0] is RuntimeError


# ---------------------------------------------------------------------------
# Lazy formatting in hot-loop debug calls
# ---------------------------------------------------------------------------

def _lj_gcmc():
    from ase import Atoms
    from ase.calculators.lj import LennardJones

    from mcpy.cell import Cell
    from mcpy.ensembles import GrandCanonicalEnsemble
    from mcpy.moves import InsertionMove, MoveSelector

    atoms = Atoms('H', positions=[[1.0, 1.0, 1.0]], cell=[6.0] * 3, pbc=True)
    cell = Cell(atoms, seed=1)
    return GrandCanonicalEnsemble(
        atoms=atoms, cells=[cell], units_type='LJ',
        calculator=LennardJones(rc=3.0, smooth=False), mu={'H': 50.0},
        species=['H'], temperature=1.0,
        move_selector=MoveSelector([1], [InsertionMove(cell, species=['H'], seed=2)],
                                   seed=3),
        random_seed=4, traj_file=None, outfile=None)


def _assert_lazy(caplog, prefix):
    """The matching record was logged with %-args (formatted only on emit),
    not a pre-built f-string."""
    recs = [r for r in caplog.records if r.msg.startswith(prefix)]
    assert recs, f'no {prefix!r} debug record'
    assert all(r.args for r in recs), 'message was formatted eagerly'


def test_acceptance_condition_debug_is_lazy(caplog):
    """Drives the real insertion acceptance path at DEBUG."""
    g = _lj_gcmc()
    with caplog.at_level(logging.DEBUG, logger='mcpy.ensembles'):
        g._acceptance_condition(0.0, 1, 100.0, 'H', 1)
    _assert_lazy(caplog, 'Lambda_db:')


def test_do_gcmc_step_debug_is_lazy(caplog):
    """Drives a real accepted insertion (mu = +50 accepts every one)."""
    g = _lj_gcmc()
    with caplog.at_level(logging.DEBUG, logger='mcpy.ensembles'):
        g.do_gcmc_step()
    assert g.n_atoms == 2
    _assert_lazy(caplog, 'Volume:')
