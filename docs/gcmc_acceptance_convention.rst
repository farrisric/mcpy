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
:mod:`mcpy.utils.set_unit_constant`):

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
- mcpy uses a *free* volume (``cell_volume * (1 - occupied_fraction)``) in
  the combinatorial factor while still sampling insertion points over the
  full cell box, so the proposal and acceptance volumes differ at high
  density.


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
matter how negative ``mu`` is — one-way accumulation decoupled from ``mu``.

The total-atom-count convention sidesteps this: as oxygen accumulates anywhere
(surface *or* subsurface), ``N_total`` grows, which suppresses further insertion.
The system is self-limiting. The buried-O irreversibility still exists (buried O
is never a deletion candidate), but it no longer drives runaway uptake.


Change history
--------------

- ``668e732`` / earlier — both ensembles born using the total atom count.
- ``adcc15b`` — switched ``GrandCanonicalEnsemble`` to the per-species count.
- ``9a15425`` — switched ``BatchedReplicaExchange`` to the per-species count.
- *this change* — reverted both back to the total atom count for reproducibility,
  and recorded the discussion here.


Where this lives in the code and tests
--------------------------------------

- ``mcpy/ensembles/grand_canonical_ensemble.py`` — ``do_gcmc_step`` passes
  ``self.n_atoms`` to ``_acceptance_condition``.
- ``mcpy/ensembles/batched_replica_exchange.py`` — ``_batched_single_move``
  passes ``r.n_atoms``.
- ``tests/test_acceptance.py`` — ``test_do_gcmc_step_feeds_total_atom_count``
  guards the convention; ``test_total_vs_perspecies_count_shifts_effective_mu``
  and ``test_perspecies_count_inserts_where_total_rejects`` quantify the gap.
- ``tests/test_custom_cell_region.py`` —
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

The textbook form above holds for ``min_insert=None``. When ``min_insert``
is set, ``MoleculeInsertionMove`` retries the random position/orientation
draw (up to 1000 times) against the cell's ``species_radii`` atoms until it
clears that minimum separation, which effectively restricts the *proposal*
to the non-overlapping volume; the *acceptance* test still divides by the
full cell volume ``V``. This is the same proposal/acceptance mismatch
already noted above for atomic moves: at high density it over-accepts
insertions by roughly :math:`V/V_\text{accessible}`.

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
