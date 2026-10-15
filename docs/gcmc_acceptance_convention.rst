GCMC acceptance: the de Broglie particle count
==============================================

This note records a deliberate, non-obvious convention in the grand-canonical
acceptance test, the discussion behind it, and the change history. It exists so
the choice is not silently "fixed" by a future contributor who (correctly)
notices it departs from the textbook form.

.. contents::
   :local:
   :depth: 1


The acceptance criterion
------------------------

Insertion and deletion are accepted with the de Broglie (grand-canonical)
probabilities (see :class:`mcpy.ensembles.GrandCanonicalEnsemble` and
:class:`mcpy.utils.SetUnits`):

.. math::

   p_\text{ins} = \frac{V}{(N+1)\,\Lambda^3}\;
                  e^{-\beta(\Delta E - \mu)},
   \qquad
   p_\text{del} = \frac{N\,\Lambda^3}{V}\;
                  e^{-\beta(\Delta E + \mu)} .

``V`` is the cell free volume, :math:`\Lambda` the thermal de Broglie
wavelength of the exchanged species, :math:`\mu` its chemical potential, and
``N`` the particle count fed to the combinatorial factor. **The question this
note settles is: which ``N``?**


Two conventions for ``N``
-------------------------

total atom count (the convention mcpy uses)
    ``N = len(atoms)`` before the move: every atom of every species, including
    the fixed substrate. This is what ``do_gcmc_step`` and
    ``BatchedReplicaExchange`` pass today (``self.n_atoms`` / ``r.n_atoms``).

per-species exchangeable count (textbook)
    ``N`` = number of atoms *of the exchanged species* inside the insertion
    region (the move's cell). This is the physically standard choice.

The physically correct quantity is the **per-species** count. The de Broglie
factor is the ideal-gas combinatorial term: the :math:`1/N!` indistinguishability
correction applies only among identical, interchangeable particles, and only to
the population that shares the insertion volume ``V``. The total atom count
mixes in distinguishable species and the fixed substrate, which do not belong in
that factor. The interaction with all those atoms enters the acceptance through
:math:`\Delta E` (the total potential energy change), not through ``N``.


Two related conventions to keep in mind:

- ``mu`` is the **full** chemical potential, including the
  ideal/:math:`\Lambda` term. So ``mu`` must be referenced on the
  calculator's absolute energy scale (e.g. an O\ :sub:`2` gas reference at
  the target ``T`` and ``p``).
- ``V`` is the cell *free* volume, and insertion points are drawn
  uniformly over that same free volume.
  The deletion uses the free volume of the state after the deletion.
  See `The volume V`_ below.


The volume V
------------

Detailed balance between an insertion and its reverse deletion needs one
condition on ``V``.
The insertion proposal density must be exactly :math:`1/V`, and the
deletion must use the same ``V``.
Both refer to the state *without* the exchanged particle.

mcpy meets this condition as follows:

- One exclusion rule, defined in :class:`mcpy.cell.Cell`.
  A point is excluded when it lies within ``species_radii[symbol]`` of any
  atom.
  The rule includes every atom, also atoms outside the region (for example
  the slab below the ``CustomCell`` floor).
  It also includes the periodic images of the atoms along every periodic
  direction.
- The free-volume estimate (``Cell.free_volume``) counts the uniform region
  points that the rule does not exclude.
- The insertion proposal (``Cell.get_random_free_point``) redraws uniform
  region points until the rule does not exclude the point.
  The result is uniform over the free volume, so its density is
  :math:`1/V`.
  ``MoleculeInsertionMove`` applies the rule to the center of mass only.
  The energy handles overlaps of the other member atoms.
- The insertion uses the cell volume that the ensemble computes for the
  current (pre-insertion) state.
- ``DeletionMove`` and ``MoleculeDeletionMove`` compute the free volume of
  the trial state after the deletion, and report it through
  ``get_volume``.
  Before ADR 0003 the deletion used the volume of the state that still
  held the deleted particle.
  That volume is too small by the particle's own exclusion sphere, which
  biased the deletion acceptance.
- A cell without ``species_radii`` excludes nothing.
  Its volume is the exact region volume and the proposal is the plain
  uniform draw.
  The periodic box ``Cell`` therefore keeps its previous behaviour and its
  previous random streams (the LAMMPS-parity runs do not change).

``min_insert`` sets a seed clearance for relaxed insertions (ADR 0004).
With it, the proposal keeps ``min_insert`` from every atom, inside the
``species_radii`` spheres too, and the acceptance still uses the
``species_radii`` free volume.
The relaxation moves the seed into that free volume.
Under unrelaxed trials this mismatch is a bias, so leave ``min_insert``
unset when the calculator does not relax.

Remaining limits:

- ``V`` is a Monte Carlo estimate with ``mc_sample_points`` points.
  Its relative noise is about :math:`\sqrt{(1-f)/(f\,n)}` for a free
  fraction ``f``.
- The rule is a hard core only for proposals.
  A displacement or a relaxation can move a particle into an exclusion
  sphere.
  A deletion can leave such a state, but an insertion cannot enter it.
  The sampling is exact only when the exclusion radii are also hard cores
  of the potential.
  With a relaxing calculator the sampling is already basin-level (ADR 0002).
- Periodic images are taken one box length in each direction, so every
  radius must be shorter than the box.

Decision records: ``docs/adr/0003-insertion-proposal-matches-free-volume.md``
and ``docs/adr/0004-min-insert-is-a-relaxation-seed-clearance.md``.


Why mcpy keeps the total atom count
-----------------------------------

**Reproducibility.** The group's entire body of GCMC work (including the
forthcoming Ag-oxidation study) was produced with the total-atom-count
convention. Switching to the per-species count shifts the *effective* chemical
potential by

.. math::

   \Delta\mu_\text{eff} = k_BT \,\ln\!\frac{N_\text{total}+1}{N_\text{species}+1},

which for a few-thousand-atom substrate with tens of exchanged atoms is
**~0.1-0.2 eV toward insertion** at typical temperatures. Concretely, after the
per-species change an oxide phase boundary moved from roughly :math:`\Delta\mu
\approx -0.5` eV to :math:`\approx -0.3` eV, and oxygen kept inserting well into
the reducing regime. Keeping the original convention keeps every simulation on
one consistent footing; the per-species count would require re-running and
re-calibrating ``mu`` across all prior work.

This is a reproducibility choice, **not** a claim that the total count is more
correct. If you start a fresh study with no need to match prior runs, prefer the
per-species count (and recalibrate ``mu`` against a gas reference).


Side effect that the total count happens to mask
------------------------------------------------

With the per-species count there is a runaway failure mode for species that
dissolve below the cell floor. ``CustomCell.get_atoms_specie_inside_cell``
excludes atoms with ``z < bottom_z`` (subsurface, "absorbed"), so such atoms are
neither counted nor eligible for deletion. Under the per-species convention this
undercount inflates :math:`V/((N+1)\Lambda^3)` and insertion stays favored no
matter how negative ``mu`` is - one-way accumulation decoupled from ``mu``.

The total-atom-count convention sidesteps this: as oxygen accumulates anywhere
(surface *or* subsurface), ``N_total`` grows, which suppresses further insertion.
The system is self-limiting. The buried-O irreversibility still exists (buried O
is never a deletion candidate), but it no longer drives runaway uptake.


Change history
--------------

- ``668e732`` / earlier - both ensembles born using the total atom count.
- ``adcc15b`` - switched ``GrandCanonicalEnsemble`` to the per-species count.
- ``9a15425`` - switched ``BatchedReplicaExchange`` to the per-species count.
- ``b97361f`` - reverted both back to the total atom count for reproducibility,
  and recorded the discussion here (``8eabe76``).
- ADR 0003 - the free volume ``V`` became one rule for insertion proposals,
  insertion acceptance and deletion acceptance (section `The volume V`_).
- ADR 0004 - ``min_insert`` came back as a seed clearance for relaxed
  insertions; the acceptance keeps the ``species_radii`` free volume.


Where this lives in the code and tests
--------------------------------------

- ``mcpy/ensembles/grand_canonical_ensemble.py`` - ``do_gcmc_step`` passes
  ``self.n_atoms`` to ``_acceptance_condition``.
- ``mcpy/ensembles/batched_replica_exchange.py`` - ``_batched_single_move``
  passes ``r.n_atoms``.
- ``tests/test_acceptance.py`` - ``test_do_gcmc_step_feeds_total_atom_count``
  guards the convention; ``test_total_vs_perspecies_count_shifts_effective_mu``
  and ``test_perspecies_count_inserts_where_total_rejects`` quantify the gap.
- ``tests/test_audit_2026_10.py``: the ``test_p1_*`` tests guard the
  volume rule, including a two-state detailed-balance check of the full
  GCMC loop (``test_p1_two_state_gcmc_obeys_detailed_balance``).
- ``tests/test_custom_cell_region.py`` -
  ``test_subsurface_oxygen_is_excluded_from_deletion_candidates`` documents the
  buried-O caveat.


Molecular moves
---------------

Rigid-molecule insertion/deletion (:class:`mcpy.moves.MoleculeInsertionMove`,
:class:`mcpy.moves.MoleculeDeletionMove`) do **not** use the total-atom
convention above. They use the textbook rigid-molecule form: ``N`` is the
number of molecules of the exchanged species whose center of mass lies inside
the move's cell, reported by the move itself through
``MoveSelector.get_exchange_count()``. :math:`\Lambda` is computed from the
total molecular mass, and ``delta_particles`` remains :math:`\pm 1` (one
molecule per move).

Orientations are sampled uniformly, so the rotational partition function is
absorbed into :math:`\mu`: the chemical potential passed for a molecular
species must be the **full** molecular chemical potential (translational,
rotational, and internal contributions of the reference reservoir included).

Atomic moves are unaffected: for them ``get_exchange_count()`` returns
``None`` and the ensembles fall back to the total-atom count documented
above.

Because molecular moves *do* use the per-species count, they are exposed to
the runaway mode described in the previous section: any escape route out of
the counted region undercounts ``N`` and inflates
:math:`V/((N+1)\Lambda^3)`. Molecule candidacy therefore goes through
``cell.is_point_exchangeable(com)``, not ``cell.is_point_inside(com)``. The
two differ for :class:`mcpy.cell.CustomCell`, whose exchangeable region drops
the z upper bound exactly as ``get_atoms_specie_inside_cell`` already does for
single atoms, so a molecule that desorbs above the cell top stays both
countable and deletable. The floor exclusion is kept on both paths: molecules
that sink below ``bottom_z`` remain excluded, and the buried-species
irreversibility noted above applies to them unchanged.
:class:`mcpy.moves.MoleculeDisplacementMove` tests the same predicate for its
one-way-door guard, so the candidacy region and the region a molecule may be
displaced within always agree.

``V`` in the molecular acceptance is the cell *free* volume, not the full
cell volume.
The center of mass is drawn uniformly over that free volume, and the
deletion uses the free volume after the molecule is removed (see
`The volume V`_).

Atomic and molecular species of the same element can coexist (e.g. atomic O
alongside O2 molecules): an atomic :class:`mcpy.moves.InsertionMove` tags its
inserted atom as free (``molecule_id = -1``, guarding against ASE's zero-fill
when extending arrays), and an atomic :class:`mcpy.moves.DeletionMove` only
picks free atoms, never molecule members. Displacement-type moves are not
molecule-aware: they can drag a member atom individually, which is
thermodynamically valid with a relaxing calculator but departs from the
rigid-molecule picture.

The MPI :class:`mcpy.ensembles.ReplicaExchange` does not support molecular
species either: its per-species swap bookkeeping counts atoms by symbol
(``atoms.symbols.count(specie)``), which is always 0 for a molecular name, so
its constructor raises ``NotImplementedError`` when the wrapped GCMC ensemble
has molecular species configured; :class:`mcpy.ensembles.BatchedReplicaExchange`
is the supported replica-exchange path for molecular GCMC.
