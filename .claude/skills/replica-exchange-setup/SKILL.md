---
name: replica-exchange-setup
description: Design or diagnose a replica-exchange ladder (temperature or chemical-potential rungs) for ReplicaExchange or BatchedReplicaExchange. Use when choosing rungs, reading swap acceptance, or when an isotherm looks flat.
---

# Set up a replica-exchange ladder

The full method lives in `docs/replica_exchange_ladder_spacing.rst`. Read it before you pick rungs.

## Steps

1. Pick the class:
   one GPU, or molecular species, means `BatchedReplicaExchange`; one MPI rank per replica means `ReplicaExchange` (atoms only, no molecules).
2. Space a mu ladder from a measured isotherm with `dmu = 1/sqrt(beta * dN/dmu)` (the doc has `acceptance_equalized_mu_ladder`).
   Uniform spacing fails: the swap exponent is `beta * dmu * dN`, and dN grows with coverage, so the high-coverage pairs stop swapping.
   Example: CO on Cu375Pd30 at 400 K over -1.8..-1.0 eV needed 29 rungs (0.064 eV at the bare end, 0.020 eV at high coverage), not 5.
3. For many replicas, set `chunk_size` on the calculator (see the `run-gpu` skill).
   Going wider raises GPU use and is often faster per step in total.
4. After the run, judge each neighbour pair by its second-half acceptance:
   `second_half = 2*cum_end - cum_mid`.
   A healthy pair is in the range 15-65 %; 0 % means the ladder is cut there.
   `BatchedReplicaExchange` logs a warning for each dead pair; for MPI `ReplicaExchange`, compute it from the outfile yourself.
5. Check each rung for convergence (the tail-half drift of N is small) before you report transition points.
   A live ladder can still be unconverged; chain runs from the last state if needed.

Done when: every neighbour pair has non-zero second-half acceptance and every rung has converged.

## Traps

- The cumulative acceptance column looks fine for dead ladders: early swaps between identical starting replicas are free and inflate it forever.
- Each `BatchedReplicaExchange(..., mus=[...])` result from before 2026-07-27 (v1.4.0) is invalid; re-run it.
- Temperature-ladder runs from before 2026-08-17 lack the de Broglie cross-term whenever N differs between rungs.
