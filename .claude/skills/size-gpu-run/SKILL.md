---
name: size-gpu-run
description: Use when sizing a GPU GCMC or batched replica-exchange run - choosing atom count, replica count, chunk_size, or exchange_interval, estimating memory or wall time, or diagnosing OOM / low GPU utilization before launch.
---

# Size a GPU run (atoms, replicas, memory, time)

Reference card: RTX 5090, 32 GB, nvalchemi 0.2.0, fp32, cuEq on, compile off.
All numbers below are measured on it with `benchmark/internal/re_batched_scaling.py`.
On other hardware or after an nvalchemi upgrade, re-run that script and trust its
numbers over this card.

## Steps

1. Fix the atom count per replica from the science, not the GPU.
   Then check it fits with the memory rules below.
2. Fix the replica count from the ladder, not the GPU:
   use the `replica-exchange-setup` skill (rung spacing sets the count; uniform spacing fails).
   Replica count is NOT memory-bound once `chunk_size` is set; it only costs time, linearly.
3. Estimate memory per chunk (relax path, medium-mpa-0 class model):
   - `peak ≈ 0.1 GiB + 1.3 MiB x (chunk_size x atoms_per_replica)` for supported oxide;
     ~0.9 MiB/atom for a bare metal NP.
   - MH-1 class model: ~8.2 MiB/atom; often forces `chunk_size=1`.
   - Hard ceilings on 32 GB: ~32k atoms bare NP, ~23k supported oxide, ~14k atoms per single
     batched forward.
4. Pick `chunk_size` = largest chunk whose estimate stays under ~80% of VRAM.
   Measured anchor: 8 replicas x 664 atoms whole-batch = 5.3 GB, 72% utilization, 6.8 s/step.
   `chunk_size=1` is many times slower; going wider is nearly free until utilization
   reaches ~75-80%, then throughput flattens.
5. Estimate wall time:
   - Single GCMC, ~3.7k-atom supported NP: ~3.7 s/step (2 moves), ~1.8 s per relaxation.
   - Batched RE: s/step grows roughly linearly with total batch atoms once the GPU is
     saturated; below saturation extra replicas are almost free
     (measured 1 -> 8 replicas at 664 atoms: 1.4 -> 6.8 s/step).
   - Total = gcmc_steps x s/step. Budget >= 10^4 steps per rung for convergence.
6. Set `exchange_interval` (default 10 GCMC steps).
   A swap attempt reuses the current energies, so it is nearly free; do not raise the
   interval to save time. 10 is fine; lower it only if rungs decorrelate faster than they swap.
7. Launch per the `run-gpu` skill (env `alchemi`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).

Done when: estimated peak < 80% VRAM, atoms per forward < 14k, and the step-time budget
covers >= 10^4 steps per rung.

## Traps

- Sizing memory by replica count: wrong. Peak follows `chunk_size x atoms`, not replicas.
- Trusting `nvidia-smi` during a long run: allocator fragmentation inflates it;
  use the env var and judge by torch allocated memory.
- Reusing pre-0.2.0 numbers (e.g. "8 x 664 atoms = 29 GB"): stale; 0.2.0 halved batched memory.
- Estimating MH-1 with the 1.3 MiB/atom rule: it is ~6x hungrier.
- To re-measure on new hardware or a new model: `benchmark/internal/re_batched_scaling.py`.
