# `min_insert` is a seed clearance for relaxed insertions

This decision replaces the `min_insert` part of ADR 0003.
The rest of ADR 0003 stays: the cell owns the `species_radii` rule, the free volume `V` uses it, and deletions use the post-deletion volume.

Trial energies are relaxed (ADR 0002), so an inserted atom does not stay where it was proposed.
The relaxation moves it into the space where it can sit, which the relaxed host-adsorbate distance sets.
That space is the `species_radii` free volume, so the acceptance keeps that `V`.
The proposed point is only a seed for the relaxation.
It must not land on top of an atom, and it does not need to keep the full host radius.

We decided that `InsertionMove` and `MoleculeInsertionMove` take `min_insert` again, as that seed clearance.
With `min_insert` set, the proposal keeps `min_insert` from every atom and periodic image (the center of mass for molecules), and the acceptance still uses the `species_radii` free volume.
Without `min_insert`, the ADR 0003 proposal applies: uniform over the `species_radii` free volume.
The two values have different jobs and are expected to differ.

Under textbook (unrelaxed) GCMC this mismatch is a bias, which is why ADR 0003 removed it.
Under relaxed trials no convention is exact, because a basin is weighted by the seeds that relax into it.
We prefer the seed clearance because it lets the relaxation reach sites that a full host radius blocks.

Evidence (`benchmark/insertion_rule_comparison.py`, O on a 6x6x4 Ag(111) slab, medium-mpa-0, 500 K, Delta mu_O = -0.3 eV, 600 steps, one seed):
`min_insert` 0.5 with radii Ag 2.14 accepted 19.1% of insertions at 999 FIRE steps per accepted insertion.
The ADR 0003 rule with the same radii accepted 12.3% at 1230.
The ADR 0003 rule with all radii at 0.5 accepted 21.5% at 898; it uses a larger `V`, which acts as a mu about kT ln(1873/818) = 0.036 eV higher.

Guarded by `test_min_insert_is_a_seed_clearance_adr_0004` in `tests/test_audit_2026_10.py`.
