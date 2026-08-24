# Over-engineering audit (ponytail-audit)

Date: 2026-08-24.
Line numbers refer to commit `cf70f65` (tip of `research/metric-capture`).
Scope: tracked files only (`mcpy/`, `docs/`, `examples/`, `tests/`, `benchmark/`, `scripts/`).
Untracked working dirs (`scratch/`, `runs/`, `private_thesis/`, `benchmark/internal/`) were not audited.

This is a complexity audit, not a code review.
Correctness, security and performance were out of scope, with two exceptions noted at the end because they were found on the way.
Nothing here has been applied.

Reviewable HTML version of the same findings: `.lavish/ponytail-audit.html` (untracked).

Net total: about **-2,735 lines across 43 findings**, **-0 dependencies** (`ase` and `scipy` both earn their place).

Tags follow the ponytail vocabulary.
`delete` = dead code or speculative feature, replacement is nothing.
`stdlib` = hand-rolled thing the standard library ships.
`native` = code doing what the platform or an existing dependency already does.
`yagni` = abstraction with one implementation, config nobody sets, layer with one caller.
`shrink` = same logic, fewer lines.

## Top three

1. `delete` `mcpy/moves/go_moves.py`, whole file, **-187**.
   Three of its classes (`PermutationMove`, `ShakeMove`, `BrownianMove`) are stale copy-based twins of the exported ones.
   The other four (`BallMove`, `ShellMove`, `BondMove`, `HighEnergyAtomsMove`) have no caller, no export and no test.
   Two doc mentions (`docs/moves.rst:109-113`, `docs/reference/moves.rst:151-152`) go with them.
2. `native` `docs/reference/*.rst`, **-688**.
   Five hand-typed API pages restate docstrings that already exist in the code.
   `sphinx.ext.autodoc` is already enabled in `docs/conf.py:23` and never used.
   Replace the bodies with `automodule` plus a short intro each.
3. `shrink` `docs/examples/*.rst`, **-400**.
   Nine pages paste the `examples/*.py` source into code blocks by hand, so every example now lives twice and drifts.
   Replace with `.. literalinclude:: ../../examples/x.py`.

## Main findings

| # | Tag | What to cut | Replacement | Lines | Path |
|---|-----|-------------|-------------|-------|------|
| 1 | delete | Whole module: 3 stale duplicates of exported moves + 4 never-referenced moves | Nothing | -187 | `mcpy/moves/go_moves.py` |
| 2 | native | Hand-written API reference duplicating every docstring | `sphinx.ext.autodoc` | -688 | `docs/reference/*.rst` |
| 3 | shrink | Example source pasted into 9 doc pages by hand | `literalinclude` | -400 | `docs/examples/*.rst` |
| 4 | delete | Second batched-RE example differing only by the `chunk_size` kwarg | A `--chunk-size` flag on `re_gcmc_batched.py` | -169 | `examples/re_gcmc_batched_chunked.py` |
| 5 | yagni | Two special-case swap rules (`_exchange_prob_T`, `_exchange_prob_mu`) for one decision | `BatchedReplicaExchange._accept_swap` already covers both ladders, and carries the de Broglie term the T-rule only half-implements | -95 | `mcpy/ensembles/replica_exchange.py:101-168` |
| 6 | shrink | Free-volume MC sampler written three times (sample, per-radius cKDTree, free fraction) | One `Cell._free_fraction(points, atoms)`; each cell keeps only its region mask | -75 | `cell/spherical_cell.py`, `cell/dome_cell.py`, `cell/custom_cell.py` |
| 7 | delete | Second MACE wrapper hand-building `AtomicData`/batch dicts; zero callers in the repo | Nothing; `MACE_F_Calculator` already uses `mace.calculators.MACECalculator` | -58 | `mcpy/calculators/mace_calculator.py` |
| 8 | shrink | RE header block written twice, once through `logger.info` and once to the outfile | Build the block once as a string, log it and write it | -40 | `replica_exchange.py:292-330,356-395` |
| 9 | shrink | `get_atoms_specie_inside_cell`: 4 copies of the same symbol-mask and str/list normalisation | One base method calling a per-cell `_inside_mask(positions)` | -40 | `mcpy/cell/*.py` |
| 10 | native | Hand-rolled extended-XYZ writer (`write_xyz`) | `ase.io.write(handle, atoms, format='extxyz')` with the extras in `atoms.info`; keep the hand version only if the per-frame cost the comment claims is re-measured | -40 | `ensembles/base_ensemble.py:263-305` |
| 11 | delete | Unused helpers `find_surface_indices`, `sphere_volume`, `total_volume`, `total_volume_with_overlap` (analytic overlap volume, superseded by MC sampling) | Nothing; `overlap_volume` survives only through one audit test, delete both | -35 | `mcpy/utils/utils.py:20-78` |
| 12 | yagni | `BaseCalculator` and `MACE_F_Calculator` are the same relax-then-energy wrapper; the second also accepts a prebuilt calculator | One wrapper with `optimizer=` and the step counters; keep `MACE_F_Calculator` as an alias | -30 | `calculators/base_calculator.py`, `calculators/mace_f_calculator.py` |
| 13 | delete | Dead knobs and methods: `BaseEnsemble(units_type=, user_tag=)`, `RandomNumberGenerator(warm_up=)`, `MoveSelector.get_operator`, `.calculate_volumes`, `.get_acceptance_ration` plus its test | Nothing. `units_type` appears only in the signature and is never read; `user_tag` is stored and never read | -30 | `base_ensemble.py:21,23`, `move_selector.py:116-160` |
| 14 | native | `DisplacementMove(constraints=[indices])` plus its own index cache, re-implementing a frozen-atom list the file's own docstring warns is fragile under insertion and deletion | `ase.constraints.FixAtoms`, which the GCMC loop already snapshots and restores | -20 | `moves/displacement_move.py:26-52` |
| 15 | delete | Duplicate `get_species()` (identical to `Cell`'s) and a repeated `self.species_radii = species_radii` that `super().__init__` already did | Inherit it | -18 | `spherical_cell.py`, `dome_cell.py`, `custom_cell.py` |
| 16 | native | Physical constants and unit factors hardcoded (Planck, Boltzmann, amu to kg, `sqrt(e)*1e10`) | `ase.units.kB`, `_hplanck`, `_amu`, `_e`; ase is already a hard dependency and `CanonicalEnsemble` already imports `ase.units.kB` | -12 | `utils/set_unit_constant.py:63-84` |
| 17 | native | Hand-built quaternion to rotation matrix | `scipy.spatial.transform.Rotation.random().as_matrix()`; scipy is already used for `cKDTree`. Pass a seeded generator to keep reproducibility | -12 | `moves/molecule_utils.py:80-88` |
| 18 | stdlib | Weighted pick built from `np.cumsum` + `np.searchsorted` + an index clamp | `random.choices(move_list, cum_weights=..., k=1)`; also drops numpy from the module | -10 | `moves/move_selector.py:60-78` |
| 19 | yagni | `exchangeable_predicate`'s `getattr(..., 'is_point_exchangeable', None) or ...` fallback | `BaseCell` now defines `is_point_exchangeable`, so plain attribute access is enough. Only pure duck-typed non-`BaseCell` cells need the fallback; decide whether that contract is still promised | -8 | `moves/molecule_utils.py:39-48` |
| 20 | yagni | `NullCell`: a whole cell class whose three methods return `0`, existing only because `BaseMove` insists on a cell | Make `BaseMove.cell` optional and have `get_volume` return `0.0`. Note `get_random_point()` returns the int `0`, which is not a point; nothing calls it | -35 | `cell/null_cell.py`, `moves/base_move.py` |

## Ensembles (1,827 lines across 5 files, the biggest subpackage)

One theme dominates.
The MC accept/reject loop and the replica-exchange bookkeeping each exist twice, once serial and once batched or MPI, kept in step by hand.
The repo already carries paired regression tests (`test_gcmc_rolls_back_a_move_that_mutates_then_reports_failure` and its `test_batched_re_` twin) that exist purely because the copies can drift.
That is the smell.
Findings 5 and 8 above belong to this group and are not counted again here.

| # | Tag | What to cut | Replacement | Lines | Path |
|---|-----|-------------|-------------|-------|------|
| E1 | shrink | `_batched_single_move` re-implements `do_gcmc_step` line for line: the arrays and constraints snapshot, the `is False or is None` sentinel check, the "returned a different Atoms object" guard, the de Broglie count fallback, and the whole accept branch (`wrap`, `n_atoms`, `E_old`, counter, volumes, `_record_minimum`), comments included | Split the serial step into `_propose(atoms)` and `_commit_or_rollback(E_new, meta)` on `GrandCanonicalEnsemble`; both loops then call the same two methods and cannot drift | -65 | `grand_canonical_ensemble.py:243-311` vs `batched_replica_exchange.py:235-304` |
| E2 | shrink | The seven-column replica row format `"{:<5} {:<10} {:<25} {:<15.6f} ..."` is typed out four times (header plus rows, in each RE class), and `summarize_states` builds a dict only to reformat the same values twice | One `_RE_ROW` format constant and one row-builder shared by both classes | -35 | `replica_exchange.py:311-330,384-395,404-418`, `batched_replica_exchange.py:525-543` |
| E3 | yagni | `CanonicalEnsemble` keeps the current energy in three places at once: `self._current_energy`, `atoms.info['key_value_pairs']['potential_energy']` (an ASE-GA convention nothing else here reads), and `self.lowest_energy` beside the base class's `_best_energy`/`_best_score` | Keep `_current_energy` and the base `_best_*`; drop the `key_value_pairs` round-trip and `lowest_energy` | -24 | `canonical_ensemble.py:88-96,112-125,140-152` |
| E4 | shrink | `_write_global_minimum` written twice; only the "find the best replica" line differs (MPI gather vs local min), the open/write/except tail is identical | One `write_global_minimum(path, atoms, energy, score, rank, logger)` helper next to `write_xyz` | -18 | `replica_exchange.py:274-300`, `batched_replica_exchange.py:176-195` |
| E5 | shrink | `_write_initial_row` re-types `write_outfile`'s format string just to emit `N/A` placeholders | Call `write_outfile` with a placeholder ratio list | -14 | `grand_canonical_ensemble.py:328-341` |
| E6 | yagni | `consolidate_logging=True` reaches for a logger by hardcoded module-path string, mutates its level for the duration of `run()`, and restores it in a `finally`: a global side effect to tidy console output | Have the replicas log their lifecycle at DEBUG and let the application choose the level, the way `mcpy/utils/logging.py` already promises | -14 | `batched_replica_exchange.py:120-171` |
| E7 | delete | `BaseEnsemble.__del__`: a third cleanup path after `finalize_run` and `__exit__`, which swallows every exception | Nothing; the context manager and `run()`'s `finally` already close the handles | -8 | `base_ensemble.py:175-181` |
| E8 | shrink | Parameters optional in the signature but mandatory in fact: `CanonicalEnsemble(optimizer=None, move_selector=None)`, either of which reaches a `TypeError` inside `relax()`/`do_mutation()`. Same for the one-caller helper `_exchange_pairs` | Make them required positional arguments; inline the helper | -8 | `canonical_ensemble.py:30,33`, `batched_replica_exchange.py:306-309` |

Kept on purpose, do not touch: the arrays and constraints rollback snapshot, the `is not atoms` identity guard, the two-tier interval and total acceptance counters, the swap-payload key stripping in `set_state`, and both ladder-degeneracy warnings.
Each encodes a bug that already happened once.

## Alchemi calculators (780 lines across 3 files)

The three-file split is right.
What is wrong is that the single-structure and batched paths were written twice and have already drifted apart, and that the FIRE and MD bootstrap ritual appears three times.

| # | Tag | What to cut | Replacement | Lines | Path |
|---|-----|-------------|-------------|-------|------|
| A1 | shrink | Single-structure relaxation re-implements the batched one: same pre-allocation, same freeze hooks, same NL bootstrap, same optimizer build | `get_potential_energy = _relax_batch([atoms])[0]`. The two have already drifted: the single path calls `opt.run` (whole-batch), the batch path steps manually and compacts. One of them is the good one; keep only it | -45 | `alchemi_f_calculator.py:116-205` |
| A2 | shrink | Bootstrap ritual written three times: pre-allocate `forces`/`energy`, `_freeze_hook_for`, `NeighborListHook`, `_build_nl`, `opt.compute`, then zero frozen forces, with the same three-line comment each time | One `_bootstrap(opt, batch, fixed, nl_hook)` in `_alchemi_common` | -25 | `alchemi_f_calculator.py:132-162,178-205`, `_alchemi_common.py:212-246` |
| A3 | delete | `run_md` exists twice, byte-identical body and docstring, in both calculators | One method on a small shared mixin; both classes already import `_run_langevin_md` | -22 | `alchemi_calculator.py:157-187`, `alchemi_f_calculator.py:323-344` |
| A4 | yagni | `_ALCHEMI_OPTIMIZERS` registry and `optimizer=` validation for a second optimizer (`fire2`) that no example, benchmark or test selects | Import `FIRE` directly; re-add the switch when something asks for FIRE2. Note `scratch/gcmc_ag111_alchemi.py` does default to `fire2`, so confirm before cutting | -8 | `alchemi_f_calculator.py:32,96-101` |
| A5 | shrink | `_write_back_positions` re-derives the frozen indices with its own inline `FixAtoms` loop; `_fixed_indices` is defined ten lines below it | Call `_fixed_indices(atoms)` | -8 | `_alchemi_common.py:149-168` |
| A6 | delete | `_make_batch` is exactly `_make_multi_batch([atoms])` | Keep the multi form only | -5 | `_alchemi_common.py:120-127` |
| A7 | shrink | `_per_graph_energies`: six lines of comment describing a scatter-reduced fallback, above a branch that only raises | Keep the raise, drop the fiction | -6 | `_alchemi_common.py:130-142` |

Not cuts, deliberately kept: `_HeadMACEWrapper`, the `wrapper.eval()` line, the cuEq import guard, the `energy_only`/`forces` pre-flight errors and the FIRE step-cap warning.
Every one of them has a documented failure it prevents (wrong head, retained second-order graph, silent truncation bias).

## Tests (3,505 lines, 18 files, no conftest)

| # | Tag | What to cut | Replacement | Lines | Path |
|---|-----|-------------|-------------|-------|------|
| T1 | shrink | There is no `tests/conftest.py`, so the same throwaway doubles are re-declared file by file: `StubCalc`, `_StubCalc`, `_UphillCalc`, `_HugeUphillCalc`, `StubGasCalculator`, `_PairedCalculator`, plus eight stub cell classes | One `conftest.py` with a `flat_calc` / `uphill_calc` / `stub_cell` fixture trio | -80 | `tests/` (5 files) |
| T2 | delete | Tests that exist only to hold dead code alive: `test_overlap_volume_matches_analytic`, `test_overlap_volume_limits`, the two `nullcell` tests, `test_moveselector_legacy_typo_alias_warns` | Nothing; they fall out for free with findings 11, 13 and 20 | -60 | `tests/test_audit_regressions.py` |
| T3 | yagni | Tests asserting on internals rather than behaviour: `test_no_sklearn_import` (guards a dependency the project never had), `test_max_attempts_constant_exists`, `test_no_rejection_loop` | The behavioural tests beside them already cover it (`test_impossible_min_insert_does_not_loop_forever`, `test_distribution_is_uniform`) | -20 | `tests/test_optimizations.py:93,127,174` |

The suite is otherwise in good shape.
`test_audit_regressions.py` reads like a grab bag but every case names a real past bug, and the acceptance, molecule and batched-RE files carry the physics that example scripts cannot check.
No coverage worth deleting was found.

## Phase diagram, examples, benchmarks

| # | Tag | What to cut | Replacement | Lines | Path |
|---|-----|-------------|-------------|-------|------|
| P1 | delete | The superseded phase-diagram entry point `analyze_phase_diagram_results` (190 lines) plus the two helpers only it uses (`free_en`, and `from_mu_to_press` which has zero callers anywhere) | Nothing; `docs/phase_diagrams.rst:104` already says "Prefer `plot_phase_diagram` for new work". Caveat: `notebooks/phase_diagram.ipynb` still calls it, so port that one cell first | -205 | `mcpy/utils/phase_diagram.py:32-231` |
| P2 | shrink | A four-flag `argparse` CLI inside the library module, exposing 4 of about 20 knobs, with no `console_scripts` entry in `pyproject.toml`, so it is reachable only as `python -m` by someone who already read the source | Drop it; the notebook and `docs/make_figures.py` call the function directly | -25 | `phase_diagram.py:541-563` |
| X1 | shrink | Every example calls `configure_logging()` at import time, between its imports, which forces about five `# noqa: E402` suppressions per file across 13 files | Call it on the first line of `main()`. `configure` is idempotent and library modules only ever `getLogger`, so nothing needs it at import time. Imports return to the top and every noqa disappears | -65 | `examples/*.py` (13 files) |
| B1 | shrink | The two tracked parity benchmarks copy-paste their scaffolding: `block_stats` is byte-identical, `parse_thermo` differs only by a dropped docstring, `run_lammps` by one env tweak, `compare` by a field width, and `ZeroCalc` is in both | One `benchmark/_parity_common.py` | -90 | `benchmark/lammps_gcmc_parity.py`, `benchmark/mace_gcmc_parity.py` |
| X2 | shrink | Examples import deep module paths (`mcpy.moves.move_selector`, `mcpy.ensembles.grand_canonical_ensemble`) even though both names are exported from the package `__init__` | Import from `mcpy.moves` / `mcpy.ensembles`; the examples then demonstrate the public API | -0 | `examples/*.py` |

## Looked bloated, is not

- The Alchemi and MACE calculator split itself. Energy-only vs relaxing, and CPU-ASE vs GPU-resident, are four genuinely different call paths. Do not merge the classes; the waste is inside them (A1 to A7).
- `chunk_ranges`, `_clamp_free_volume`, `molecule_com`. Small, single-purpose, each with a named failure it prevents (a `0.0` volume turning the deletion prefactor into `inf`; a molecule split across a periodic face).
- The optional-backend `try/except ImportError` in `calculators/__init__.py`. Looks like defensive noise; it is what keeps the torch-free CI matrix working.
- Narrative `docs/*.rst` vs `docs/reference/*`. Genuinely different registers (concepts vs signatures), not duplication. Only the reference half should become autodoc.

## Out of scope, found anyway

Neither is a cut, so neither is counted above.

- `scipy` is imported by three cell modules (`cKDTree`) but is not in `pyproject.toml` dependencies. It currently arrives only as an ase or mace transitive.
- `analyze_phase_diagram_results` carries about 20 Ag/O-specific magic defaults in its signature (`e_host=-2.82894684`, `idx_ref=2400`, `z_threshold=13.4`, `output_plot_path="lines_phases_mace.png"`). Any other system silently gets wrong physics rather than an error. If P1 is not applied, make those parameters required.

## Suggested order

Lowest risk first, and each step is independently shippable.

1. Findings 1, 7, 11, 13, 15, T2: pure dead code and dead parameters. No behaviour change, about -350 lines.
2. Findings 2, 3, X1, B1, P2: docs and script hygiene. No library change, about -1,270 lines.
3. E1, then E2, E4, E5: the ensemble duplication, guarded by the existing paired regression tests.
4. A1 to A7: the Alchemi internals, guarded by `benchmark/internal/verify_compact_parity.py`.
5. Findings 5, 6, 9, 10, 16, 17, 18, 20: the shared-helper extractions and the native or stdlib swaps. These change behaviour at the margins (RNG streams, float formatting), so they need the golden regression test in place first.
