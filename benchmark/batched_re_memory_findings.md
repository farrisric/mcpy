# Batched RE GPU-memory floor — findings (2026-06-09)

How far can peak GPU memory be cut in batched RE-GCMC energy evaluation
(`AlchemiCalculator.get_potential_energies`) without changing sampled energies
beyond the calculator's run-to-run noise? Swept cuboctahedral/octahedral Ag NPs,
unsupported, over `(atoms_per_replica, n_replicas)` on an RTX 5090 (32 GB),
`medium-mpa-0`, fp32, cuEQ on. Data: `batched_re_memory.csv`,
`batched_re_memory_summary.md`, `batched_re_memory_heatmap.png`.

## TL;DR

Two bit-exact levers matter; one does not.

- **Chunking (`chunk_size`) is the win.** Peak memory is set by the largest
  *chunk*, fully decoupled from replica count. Replica count becomes
  time-bounded, not memory-bounded.
- **`energy_only` is a flat ~12 % bonus**, bit-exact, essentially free.
- **`max_neighbors` is inert here** — the neighbour-list tensor is not the
  bottleneck. Don't bother tuning it.

## 1. Noise floor (the exactness bar)

True IEEE bit-exactness is unattainable: cuEQ + scatter/atomicAdd are
non-deterministic run-to-run, and the absolute jitter scales with energy
magnitude (system size). Per-size, config-matched (same size, same `n`,
whole-batch rerun) floor:

| atoms/replica | noise floor (eV) |
|---|---|
| 256  | ~1e-4 |
| 1024 | ~1e-3 |
| 2048 | ~5–9e-3 |
| 3584 | ~6e-3–1.6e-2 |

This is the calculator's intrinsic nondeterminism, not something the levers
introduce. (At 300 K, β·1e-2 eV ≈ 0.4 — non-trivial for a single acceptance, but
inherent to fp32+cuEQ; fp64 or torch deterministic algorithms would shrink it at
a memory/speed cost — out of scope.)

## 2. Baseline surface

Baseline peak memory (forces on, whole batch), MB:

| atoms/replica \ n | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 256  | 545  | 986  | 1869  | 3636 |
| 1024 | 2131 | 4160 | 8212  | 16320 |
| 2048 | 4351 | 8594 | 17085 | OOM |
| 3584 | 7747 | 15394| OOM   | OOM |

≈ **2.08 MB per total atom**, linear in `n_replicas × atoms_per_replica`. OOM
appears past ~14 k total atoms. So whole-batch RE hits the ceiling via replica
count just as fast as via per-replica size.

## 3. Lever A — chunking (dominant, bit-exact)

`chunk_size=k` evaluates the replicas in sub-batches of `k`, so peak memory
equals the baseline of a `k × atoms_per_replica` batch — **independent of how
many replicas there are**:

| config | whole-batch | chunk=1 |
|---|---|---|
| 1024 × 8 | 16320 MB | **2136 MB** (7.6×) |
| 2048 × 8 | OOM | **4354 MB** |
| 3584 × 8 (28.7 k atoms) | OOM | **7755 MB** |

**Bit-exact.** MACE message passing is per-graph, so chunking cannot change a
replica's energy. Empirically, the chunk-vs-baseline deviation tracks the
run-to-run noise floor, shows no dependence on `chunk_size`, and exceeds the
config-matched rerun floor only ~half the time — exactly what you expect if the
added error is zero and you are comparing two single noise draws.

**Cost:** wall time grows with the number of chunks (more, smaller kernel
launches; e.g. 3584 × 8 at chunk=1 = 8 sequential forwards). For small replicas
at chunk=1 the GPU is underutilised, so chunk to the largest size that fits
rather than all the way to 1.

## 4. Lever C — energy_only (small, bit-exact, free)

Dropping `'forces'` from the model's `active_outputs` (constructor flag
`energy_only=True`) sets `compute_force=False`, so no autograd graph is built.
MC energy evaluation never uses forces.

- Flat **~88 % of baseline peak** (≈12 % saving) across every size and replica
  count.
- Bit-exact: deviations at the noise-floor level.

(The originally planned `torch.inference_mode` approach FAILS — the MACEWrapper
computes forces via autograd eagerly inside `forward`. The config flag is the
correct mechanism.)

## 5. Lever B — max_neighbors (no effect; negative result)

Tightening `max_neighbors` from `None` down to 16 changes peak memory by
**< 0.1 %** at every size (e.g. 3584 atoms: 7747 → 7747 MB), and leaves energies
unchanged within noise. The actual neighbour count within the model cutoff is
≤ 16 for these NPs, so the cap never binds — and the neighbour-list tensor is not
the memory bottleneck regardless (model activations / edge features dominate).
**`max_neighbors` is not a useful memory lever in the nvalchemi path.**

## 6. Combined answer + confirmation

`chunk_size=1` + `energy_only=True` ⇒ peak memory ≈
`0.88 × 2.08 MB × atoms_per_replica`, **independent of replica count**:

- 1024-atom NP: ~1.9 GB/replica → dozens of replicas on 32 GB.
- 3584-atom NP: ~6.8 GB → was OOM at 4 replicas whole-batch; now any replica
  count fits.

**Confirmation run** (`confirm_batched_re_safe.py`): 16 replicas × 1150-atom Ag
NP = 18.4 k atoms. Whole-batch energy eval **OOMs**; with `chunk_size=1` +
`energy_only` the full `BatchedReplicaExchange` (10 GCMC steps, exchanges) runs
at **3117 MB**, energies finite, O adsorption observed — clean end-to-end.

**Bottom line:** the replica-count memory ceiling is removed by chunking
(7–10× lower peak, OOM configs run), `energy_only` adds a free ~12 %, and
`max_neighbors` is a dead end. Memory is no longer the constraint on replica
count — wall time is.

## 7. Relaxation path (`AlchemiFCalculator`)

`chunk_size` was also added to the batched-FIRE relaxation calculator
(`get_potential_energies`). `energy_only` does not apply — a relaxer needs forces.

- Memory: 4 × 483-atom NPs, whole-batch 3660 MB → chunk=1 **992 MB (27 %)**; same
  per-largest-chunk decoupling as the energy-only path.
- Exactness: **not bit-exact.** Whole-vs-whole relaxation reproducibility floor
  is 3.7e-4 eV, but chunk-vs-whole differs by 6.5e-3 eV (~1e-5 eV/atom) — batched
  FIRE shares trajectory state (timestep/stopping), so chunking perturbs the
  path. Both converge to the same minimum within `fmax=0.05`; the gap shrinks
  with tighter `fmax`. So: **converged-equivalent, not bit-identical.**
