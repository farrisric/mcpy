# Benchmarks: mcpy GCMC vs LAMMPS `fix gcmc`

mcpy's grand-canonical sampler, for both atomic and rigid-molecule moves, has been cross-validated against LAMMPS.
Two rerunnable campaigns live in this directory; the tables below are their verdicts.

## 1. Lennard-Jones parity (`lammps_gcmc_parity.py`)

Setup: LJ units, T* = 2.0, cubic box L = 9 sigma, truncated-and-shifted lj/cut 3.0 matched between codes to <= 1e-8 by a two-body probe.
Block-averaged statistics (15 blocks, 40% burn-in); pass criteria |d<N>| < 2 sigma and |d<PE>| < 3 sigma.

Stage 0, ideal gas vs the analytic grand-canonical result (<N> = 98.66):

| code | <N> | deviation |
|---|---|---|
| mcpy | 99.32 ± 1.60 | 0.41 sigma |
| LAMMPS | 98.52 ± 0.83 | 0.17 sigma |

Stage 1, atomic LJ isotherm (insert + delete + translate):

| mu | <N> mcpy | <N> LAMMPS | d (sigma) |
|---|---|---|---|
| -3.0 | 217.85 ± 3.37 | 219.24 ± 1.03 | 0.40 |
| -2.4 | 275.31 ± 2.55 | 279.60 ± 1.78 | 1.38 |
| -1.8 | 330.17 ± 1.74 | 331.00 ± 1.55 | 0.36 |

Stage 2, rigid LJ dimer exchanged as whole molecules (mcpy `MoleculeInsertionMove`/`MoleculeDeletionMove` vs LAMMPS `fix gcmc mol`, exchange-only so the proposal distributions match exactly):

| mu | <N_mol> mcpy | <N_mol> LAMMPS | d (sigma) |
|---|---|---|---|
| -6.0 | 73.23 ± 1.43 | 72.48 ± 0.97 | 0.43 |
| -5.0 | 147.51 ± 2.74 | 145.81 ± 2.00 | 0.50 |
| -4.0 | 193.41 ± 1.39 | 191.47 ± 1.34 | 1.00 |

`<PE>` agrees at every point as well (0.4-1.4 sigma).
**All stages pass.**

Rerun: `python benchmark/lammps_gcmc_parity.py --stage all` (needs a LAMMPS binary with the MC and MOLECULE packages, e.g. conda-forge `lammps`; pass it with `--lmp`).

## 2. MACE parity (`mace_gcmc_parity.py`)

Same protocol with the MACE-MPA-0 potential (metal units, T = 400 K), against a LAMMPS build with `pair_style mace` (ACEsuit fork) using the identical exported model.

| check | result |
|---|---|
| Single-point energies (Ag2O probe; Ag slab + O2 adsorbed and beyond cutoff) | agree to 1e-5 eV (1e-9 in float64) |
| Ideal rigid-O2 gas, physical de Broglie wavelength (mass 32) vs analytic | both codes <= 1 sigma |
| LAMMPS region + `full_energy` machinery vs analytic (zero potential) | pass |
| Interacting O2/Ag(111) GCMC | disagrees, isolated to a LAMMPS-side defect |

The interacting disagreement was dissected to a defect in the LAMMPS-MACE fork's `fix gcmc` trial-insertion path, not in mcpy:
LAMMPS's own deletion acceptance matches textbook theory exactly (20.3% measured vs predicted) while its insertions are ~5x suppressed and insensitive to mu, consistent with the inserted molecule's intramolecular edge missing from the trial-energy graph, and the same code path segfaults when started from an empty or near-empty box.
mcpy evaluates every trial with a freshly built graph by construction, so this failure mode cannot occur.

Net result: **mcpy reproduces LAMMPS everywhere LAMMPS's own machinery is sound**, and mcpy's MACE GCMC is validated by composition (sampler proven on LJ, potential proven pointwise against `pair_style mace`).

Rerun: `python benchmark/mace_gcmc_parity.py --lmp <patched lmp> --model <model-lammps.pt> --python-model <mace model>` (the ACEsuit LAMMPS build recipe and full forensic notes are kept with the internal material).

## Internal material

Profiling, memory-scaling studies, raw run outputs, and detailed campaign notes are deliberately kept out of the public tree (`benchmark/internal/`, git-ignored); ask the maintainers if you need them.
