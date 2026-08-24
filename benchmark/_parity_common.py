"""Shared scaffolding for the LAMMPS parity benchmarks.

``lammps_gcmc_parity.py`` (LJ units) and ``mace_gcmc_parity.py`` (metal units,
MACE) run the same comparison against LAMMPS ``fix gcmc``, so they need the same
four things: drive LAMMPS on a generated deck, parse its thermo output, block
average a series, and print a pass/fail line. Both had their own copy;
``block_stats`` was byte-identical and the rest differed only by a field width,
a dropped docstring, and one environment variable.
"""
import math
import os
import subprocess

import numpy as np


class ZeroCalc:
    """Zero energy, for the ideal-gas stage where only the prefactor matters."""

    def get_potential_energy(self, atoms):
        return 0.0


def run_lammps(deck, outdir, tag, lmp, extra_env=None):
    """Write ``deck``, run ``lmp`` on it, and return the parsed thermo output.

    ``extra_env`` is merged over ``os.environ`` for the child; the MACE
    benchmark uses it to put libtorch on ``LD_LIBRARY_PATH``.
    """
    deckfile = os.path.join(outdir, f'in.{tag}')
    logfile = os.path.join(outdir, f'log.{tag}')
    with open(deckfile, 'w') as f:
        f.write(deck)
    env = None
    if extra_env:
        env = dict(os.environ)
        env.update(extra_env)
    subprocess.run([lmp, '-in', os.path.abspath(deckfile),
                    '-log', os.path.abspath(logfile), '-screen', 'none'],
                   check=True, cwd=outdir, env=env)
    return parse_thermo(logfile)


def parse_thermo(logfile):
    """Collect thermo rows from every run section. Returns dict of arrays
    keyed by lower-cased column names (step, atoms, pe, ...)."""
    cols, rows = None, []
    with open(logfile) as f:
        in_section = False
        for line in f:
            s = line.split()
            if not s:
                continue
            if s[0] == 'Step':
                cols = [c.lower() for c in s]
                in_section = True
                continue
            if in_section:
                if s[0] == 'Loop' or line.startswith('WARNING'):
                    in_section = False
                    continue
                try:
                    rows.append([float(v) for v in s])
                except ValueError:
                    in_section = False
    if cols is None or not rows:
        raise RuntimeError(f'no thermo data parsed from {logfile}')
    rows = [r for r in rows if len(r) == len(cols)]
    data = np.array(rows)
    out = {c: data[:, i] for i, c in enumerate(cols)}
    if 'poteng' in out:  # thermo header prints "PotEng" for pe
        out['pe'] = out['poteng']
    return out


def block_stats(series, burn_frac=0.4, nblocks=15):
    """Block-averaged mean and standard error after discarding ``burn_frac``."""
    x = np.asarray(series, dtype=float)
    x = x[int(len(x) * burn_frac):]
    blocks = np.array_split(x, nblocks)
    means = np.array([b.mean() for b in blocks])
    return means.mean(), means.std(ddof=1) / math.sqrt(nblocks)


def compare(label, m_mcpy, s_mcpy, m_lmp, s_lmp, nsig, width=10):
    """Print one pass/fail line and return ``(ok, row)``.

    ``ok`` is ``|difference| < nsig * sigma_combined``; ``row`` is the tuple the
    CSV writer expects. ``width`` sizes the mean columns (metal-units energies
    need more room than LJ ones).
    """
    diff = abs(m_mcpy - m_lmp)
    scomb = math.sqrt(s_mcpy ** 2 + s_lmp ** 2)
    ok = diff < nsig * scomb
    sigmas = diff / scomb if scomb > 0 else float('inf')
    print(f'    {label:6s} mcpy {m_mcpy:{width}.4f} ± {s_mcpy:.4f}   '
          f'lammps {m_lmp:{width}.4f} ± {s_lmp:.4f}   '
          f'|d|={diff:.4f} ({sigmas:.2f} sigma)  '
          f'{"PASS" if ok else "FAIL"}')
    return ok, (m_mcpy, s_mcpy, m_lmp, s_lmp, diff, scomb)


def demo():
    """Self-check: block_stats on a known series, compare's verdict logic."""
    mean, err = block_stats(list(range(100)) * 2, burn_frac=0.0, nblocks=10)
    assert abs(mean - 49.5) < 1e-9, mean
    assert err > 0
    ok, row = compare('demo', 1.0, 0.1, 1.05, 0.1, 2)
    assert ok and len(row) == 6
    ok, _ = compare('demo', 1.0, 0.01, 2.0, 0.01, 2)
    assert not ok
    print('ok')


if __name__ == '__main__':
    demo()
