# Metric-capture mechanisms for the batched-RE regression benchmark

Research ticket: [#27](https://github.com/farrisric/mcpy/issues/27), child of wayfinder map [#23](https://github.com/farrisric/mcpy/issues/23).
Date: 2026-08-18.
Line numbers refer to commit `918ac54` (branch point of `research/metric-capture`).

## Question

How should the harness capture peak allocated GPU memory, a GPU-utilization figure, per-step wall time split into relaxation vs overhead, and per-replica relax iteration counts, without perturbing the timings it measures?

## Recommendations (one mechanism per metric)

### (a) Peak allocated GPU memory: `torch.cuda.reset_peak_memory_stats()` after warmup + `torch.cuda.max_memory_allocated()` at the end

This is the established pattern in `benchmark/internal/re_batched_scaling.py`.
The peak counter is reset before the run at line 173 (after `torch.cuda.empty_cache()` at line 172) and reset again inside the timed step wrapper at line 167 once warmup completes, so first-step allocation and compile transients are excluded.
The single read happens once per config at line 189 (`peak_torch = torch.cuda.max_memory_allocated() / 1024 ** 2`).
`remeasure_supported_np.py:54-61` uses the same reset-then-read pattern per relaxation.
Use allocated, never reserved: `gcmc_memory_growth.py:5-14` documents that reserved tracks the caching-allocator pool, which never shrinks and grows with fragmentation, and the repo's history shows earlier reserved-metric numbers were misleading.
Reading a counter costs nothing on the timed path, so this mechanism is perturbation-free by construction.
Nothing is missing; the harness copies the `re_batched_scaling.py` pattern verbatim.

### (b) GPU utilization: the threaded nvidia-smi sampler (`GpuSampler`), windowed to post-warmup

`re_batched_scaling.py:64-93` already defines a `GpuSampler` thread that polls `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits` via subprocess every 0.15 s and stores `(perf_counter, mem_MB, util_pct)` tuples.
The post-warmup window is selected at line 192 (`sampler.window(warm_done_t[0])`) and mean/max utilization are computed at lines 194-196.
The sampler runs in its own thread and subprocess, never touches the CUDA stream, and its CPU cost is negligible, so it does not perturb the timings.
It also yields whole-device memory as a free cross-check of the allocated figure (the docstring at lines 9-10 notes smi includes the CUDA context and resident model).
No pynvml or `torch.cuda.utilization` is used anywhere in the repo; introducing one would add a dependency for no gain.
Nothing is missing; lift `GpuSampler` into the harness and report the post-warmup mean (with max as a secondary column if wanted).

### (c) Per-step wall time split into relaxation vs overhead: synced `perf_counter` brackets at two nesting levels

The house timing pattern is `time.perf_counter()` before, `torch.cuda.synchronize()` immediately before the stop read, and warmup steps excluded from the aggregate.
It appears in `re_batched_scaling.py:161-168` (step wrapper), `benchmark_alchemi_vs_mace.py:69-82` (`time_call`), and `remeasure_supported_np.py:54-60`.
For the total step time, wrap `BatchedReplicaExchange._batched_gcmc_step` exactly as `re_batched_scaling.py:159-170` already does; the method (defined at `mcpy/ensembles/batched_replica_exchange.py:208-233`) has no timing of its own.
For the relaxation share, wrap `calculator.get_potential_energies`, which is the single funnel every batched relaxation passes through: the contract is enforced at `batched_replica_exchange.py:98-102` and the call sites are `_rebatch_initial_energies` (line 202) and `_batched_single_move` (line 278).
Overhead is then total step time minus the summed relax time inside that step; no third timer is needed.
The added `torch.cuda.synchronize()` at the end of the relax bracket is effectively free because the calculator already forces a device-to-host copy to return Python floats; the sync only moves the wait to a named point.
Do not rely on the library's own `_last_step_seconds` (`mcpy/ensembles/grand_canonical_ensemble.py:356-359`): it is unsynced, only reaches the console logger, and stays 0.0 under batched RE because `_batched_gcmc_step` bypasses `GrandCanonicalEnsemble._run` entirely (`batched_replica_exchange.py:228-233`).
Missing piece: only the two harness-side wrappers themselves; no library change is required, and keeping the timing in the harness keeps the production path unperturbed.

### (d) Per-replica relax iteration counts: record graph retirement steps in `_run_compacted`

The existing metric surface is the `last_relax_steps` / `total_relax_steps` counter pair on `AlchemiFCalculator` (`mcpy/calculators/alchemi_f_calculator.py:95-96`), set from `opt.step_count` on the single-graph path (lines 163-164) and aggregated across chunks at lines 315-316.
`remeasure_supported_np.py:64,128` and `benchmark_alchemi_vs_mace.py:148-159` already consume these counters.
They are not per-replica in batched mode: `last_relax_steps` becomes the per-chunk max and `total_relax_steps` a sum, because nvalchemi's `BaseDynamics.step_count` is batch-global (`nvalchemi/dynamics/base.py:1455,1950`).
The per-graph signal does exist transiently: `_run_compacted` steps FIRE manually and receives the converged-graph indices each iteration (`batch, conv = opt.step(batch)` at `alchemi_f_calculator.py:231`, `n_steps += 1` at 232), then retires those graphs at lines 240-242 without recording when each one converged.
`BaseDynamics.step` returning the converged-index tensor (`nvalchemi/dynamics/base.py:1865`) is the only per-graph iteration signal nvalchemi exposes on the paths mcpy uses, so no upstream plumbing is needed.
Missing piece: a small addition to `AlchemiFCalculator` that appends `(original graph row -> n_steps)` at the existing retirement point into a `last_relax_steps_per_graph` list (plus the cap-hit case for graphs that never converge), which the harness reads after each step and maps back to replicas via the chunk layout.
The bookkeeping is a few integer appends at retirement points where CPU-side work (`_harvest`, lines 257-271) already happens, so it does not perturb the timed path.
If windowed reporting is wanted, follow the house counter style of dual interval-plus-cumulative tallies with a reset method, as in `MoveSelector` (`mcpy/moves/move_selector.py:66-71,138-165`).

## Inventory of existing measurement code

### `benchmark/internal/re_batched_scaling.py`

Peak allocated memory via `reset_peak_memory_stats` (lines 167, 173) and `max_memory_allocated` (line 189), with resident model-plus-context memory captured separately via `memory_allocated()` after a warm forward (line 236) and attached to every row (line 245).
Whole-device memory and utilization via the threaded `GpuSampler` nvidia-smi poller (lines 64-93), windowed post-warmup (lines 192-196).
Step timing via a monkey-patched `_batched_gcmc_step` wrapper with `synchronize()` before the stop read (lines 159-168) and warmup exclusion (line 197, default warmup 2 at line 383).
OOM configs are caught and recorded as `status='oom'` (lines 180-183).
No relax-iteration counting and no alloc-conf handling.

### `benchmark/internal/gcmc_memory_growth.py`

Per-step instantaneous triple read after `synchronize()` (lines 84-91): `memory_allocated`, `memory_reserved`, and a one-shot nvidia-smi helper (lines 41-48).
Its docstring (lines 5-14) is the canonical allocated-vs-reserved pitfall text and documents running under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set in the environment.
End-of-run `empty_cache()` probe distinguishes fragmentation from a leak (lines 99-106).
No timing, no utilization, no iteration counting.

### `benchmark/internal/remeasure_supported_np.py`

Per-relax `synchronize(); empty_cache(); reset_peak_memory_stats()` then reads of both `max_memory_allocated` and `max_memory_reserved` (lines 50-65, repeated for the batched point at lines 96-104).
Relax timing is median-of-3 `perf_counter` with pre-stop `synchronize()` (lines 57-60, 69) and explicit uncounted compile warmups (lines 87, 91).
GCMC throughput is aggregate: one `perf_counter` pair around `gcmc.run(GCMC_STEPS)` divided by step count (lines 123-127), which cannot split relax from overhead.
Reads `calc.last_relax_steps` per relax (line 64) and `calc.total_relax_steps` after the run (line 128).

### `benchmark/internal/benchmark_alchemi_vs_mace.py`

Generic `time_call(fn, n_warmup, n_repeats)` helper: untimed warmups, then `sync(); t0; fn(); sync(); append` (lines 69-82), aggregated as a mean (lines 116-117, 153-154).
Resets `total_relax_steps = 0` per system and averages over warmup-plus-repeats calls (lines 148-149, 157-159); note the warmup relaxations are included in the step average even though excluded from timing.
No memory or utilization capture.

### `mcpy/ensembles/`

`BaseEnsemble` initializes `_last_step_seconds` (`base_ensemble.py:51`); `GrandCanonicalEnsemble._run` fills it unsynced and logs it to the console only, on outfile-interval steps (`grand_canonical_ensemble.py:356-365`).
`BatchedReplicaExchange` has no timing, no memory, and no iteration instrumentation; `_batched_gcmc_step` (lines 208-233) is bare and calls each replica's `write_outfile` directly, bypassing `_run`.
`write_outfile` rows carry step, atom counts, energy, and move-acceptance ratios only (`grand_canonical_ensemble.py:143-176`; `batched_replica_exchange.py:525-543`).
The exchange counters with midpoint snapshots (`batched_replica_exchange.py:113-119,148-150`) are a ready-made template for windowed metrics.
No ensemble reads `last_relax_steps` or `total_relax_steps`.

### `mcpy/calculators/`

`AlchemiFCalculator` holds the only relax-iteration counters (details in recommendation d).
`MACE_F_Calculator` mirrors the counter pair from ASE `opt.nsteps` (`mace_f_calculator.py:27-28,54-55`).
`BaseCalculator` discards its optimizer and exposes no counters (`base_calculator.py:14-22`).
`AlchemiCalculator` is energy-only with no relaxation or instrumentation; `energy_only=True` drops forces so no autograd graph is built (`alchemi_calculator.py:78-83`).
`_alchemi_common.py` imports `DynamicsContext` and the freeze/NaN hooks (lines 16, 19) but records no metrics; a custom `AFTER_STEP` or `ON_CONVERGE` hook receiving `DynamicsContext` (`nvalchemi/hooks/_context.py:60`, carrying `step_count` and `converged_mask`) is a viable alternative capture point for (d), but the `_run_compacted` retirement point is simpler because the loop already owns `n_steps` and the compaction bookkeeping.

### Orphan data note

`benchmark/internal/relax_step_spread.csv` and `relax_step_spread_smoke.csv` exist but no script in the repo references them; the script that produced per-graph relax-step spread data appears to have been deleted.
