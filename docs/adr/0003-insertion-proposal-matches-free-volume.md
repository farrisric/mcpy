# Insertion proposal and acceptance use one free volume

The GCMC acceptance divides by the cell free volume `V`.
For exact detailed balance the insertion proposal density must be exactly `1/V`, and the reverse deletion must use the same `V`, both for the state without the exchanged particle.
Before this decision three things broke that condition: insertions redrew against a per-move scalar `min_insert` that the volume estimate did not know, that redraw saw only atoms inside the region (no slab below the floor), and deletions used the volume of the state that still held the deleted particle.

We decided that the cell owns one exclusion rule: a point is excluded within `species_radii[symbol]` of any atom, periodic images included.
`Cell.free_volume` estimates `V` with that rule, and `Cell.get_random_free_point` redraws insertion points with the same rule, so the proposal is uniform over the same `V`.
Deletion moves compute the free volume of the post-deletion trial state (about 46 ms for 3.7k atoms, small next to a relaxation).
Molecule insertions apply the rule to the center of mass.
A cell without `species_radii` excludes nothing and keeps its exact region volume, so the periodic box `Cell` and the LAMMPS-parity runs are unchanged.

`min_insert` is deprecated and ignored with a `FutureWarning`, because a per-move radius cannot be consistent with a per-cell volume that the deletion moves and the ensemble also use.
Users keep the old clearance by setting `species_radii` of at least the old `min_insert` for every species.

We rejected the alternative of proposing over the full region volume and rejecting overlaps as zero-acceptance trials.
It is also exact and needs no estimate, but it changes the volume in the acceptance from the free volume to the region volume and wastes trials at high density.

Limits: `V` carries Monte Carlo noise; the exclusion is a hard core for proposals only, so states reached by displacement or relaxation inside an exclusion sphere can be left by deletion but not entered by insertion.
Full discussion: `docs/gcmc_acceptance_convention.rst`, section "The volume V".
Guarded by the `test_p1_*` tests in `tests/test_audit_2026_10.py`.
