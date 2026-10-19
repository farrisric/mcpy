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
    the fixed substrate. This is what ``_commit_or_rollback`` passes today
    (``self.n_atoms``). ``GrandCanonicalEnsemble`` and
    ``BatchedReplicaExchange`` both score their trials with it.

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
Under unrelaxed trials this mismatch is a bias: it shifts :math:`\mu` by
:math:`k_BT \ln(V_\mathrm{free}/V_\mathrm{seed})`.
Leave ``min_insert`` unset when the calculator does not relax.
``GrandCanonicalEnsemble`` warns (``UserWarning``) when a move sets
``min_insert`` and the calculator has no relaxation steps.

Remaining limits:

- ``V`` is a Monte Carlo estimate from ``n`` uniform points.
  Its relative noise is about :math:`\sqrt{(1-f)/(f\,n)}` for a free
  fraction ``f``.
  ``mc_sample_points`` is the maximum ``n``.
  The first estimate uses all of them, and each later estimate picks ``n``
  from the previous ``f`` so that the noise is about 0.5%.
- The rule is a hard core only for proposals.
  A displacement or a relaxation can move a particle into an exclusion
  sphere.
  A deletion can leave such a state, but an insertion cannot enter it.
  The sampling is exact only when the exclusion radii are also hard cores
  of the potential.
  For molecules the rule tests the center of mass only.
  An insertion can put an atom of molecule B within ``r`` of the center of
  mass of molecule A, and then the deletion of A has no reverse insertion.
  So for molecules the potential must forbid any atom within ``r`` of the
  center of mass of another molecule.
  Between atoms, that is a hard core of ``r`` plus the largest distance of a
  member atom from its center of mass.
  This holds even without displacement moves or relaxation.
- With a relaxing calculator the sampling is basin-level (ADR 0002), and
  the effective chemical potential differs from ``mu``: see
  `Relaxed trials shift the effective mu`_.
- Periodic images are taken one box length in each direction, so every
  radius must be shorter than the box.

Decision records: ``docs/adr/0003-insertion-proposal-matches-free-volume.md``
and ``docs/adr/0004-min-insert-is-a-relaxation-seed-clearance.md``.


Relaxed trials shift the effective mu
-------------------------------------

With relaxed trials (ADR 0002), every insertion point in the catchment of
a basin ``j`` relaxes into that basin.
The catchment volume :math:`v_j` is the part of the free volume ``V``
whose points relax into ``j``.
Assume that a deletion from :math:`S+j` relaxes back to :math:`S`.
Detailed balance between the two states then gives

.. math::

   \frac{\pi(S+j)}{\pi(S)} =
   \frac{v_j\,(N_s+1)}{(N+1)\,\Lambda^3}\;
   e^{\beta(\mu - \varepsilon_j)},
   \qquad
   \varepsilon_j = E(S+j) - E(S).

:math:`N_s` is the number of deletion candidates of the species in
:math:`S`, and ``N`` is the count of the acceptance rule.
For atoms ``N`` is the total atom count; for molecules :math:`N = N_s`,
and the two counts cancel.
``V`` cancels too: the proposal density :math:`1/V` and the acceptance
factor ``V`` remove each other.

A phase diagram or a computational hydrogen electrode (CHE) analysis gives
each state the weight :math:`e^{\beta(\mu - \varepsilon_j)}` instead.
Relative to that axis, the run samples at :math:`\mu + \delta` with

.. math::

   \delta = k_BT \ln\frac{v_j}{\Lambda^3}
           + k_BT \ln\frac{N_s+1}{N+1}.

The first term is positive for any catchment larger than :math:`\Lambda^3`.
The second term is negative with the total atom count, and zero for
molecules.

Worked numbers, for O on the Ag(111) slab of :doc:`getting_started`
(4x4x3 slab, a 5 Å ``CustomCell`` that starts 0.5 Å above the top layer,
radii Ag 2.14 Å and O 0.5 Å, 500 K):

- ``V`` is about 442 Å\ :sup:`3`.
  If the 32 hollow sites (fcc and hcp) share it equally,
  :math:`v_j \approx 14` Å\ :sup:`3`.
- :math:`\Lambda_\mathrm{O} = 0.195` Å, so the volume term is +0.32 eV.
- With ``N`` = 48 + :math:`N_\mathrm{O}`, the count term is -0.17 eV at
  :math:`N_\mathrm{O} = 0`, -0.08 eV at 8 and -0.06 eV at 16.
- So :math:`\delta` is +0.16 eV on the clean slab, +0.24 eV at 8 O and
  +0.27 eV at 16 O.

For H on the Pt\ :sub:`55` particle of ``examples/05_pt_nanoparticle_h_che.py``
at 300 K, :math:`\Lambda_\mathrm{H} = 1.0` Å.
The volume term is then small (+0.04 eV for :math:`v_j = 5` Å\ :sup:`3`),
and the count term is -0.10 eV on the bare particle.

:math:`\delta` is not a constant.
It depends on the cell height and ``species_radii`` (through :math:`v_j`),
on the temperature (through :math:`k_BT` and :math:`\Lambda`), and on the
optimizer and its settings (which seeds reach which basin).
With the total atom count it also depends on the coverage and on the size
of the substrate.

Thus the :math:`\Delta\mu` (or the electrode potential ``U``) of a run or
of a replica-exchange rung is a nominal value.
Do not read a phase boundary from the rung where the coverage jumps.
Read it from ``plot_phase_diagram`` (:doc:`phase_diagrams`), which
computes the free energy of each frame from its energy and composition
only, on the :math:`e^{\beta(\mu - \varepsilon)}` axis.


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

- ``mcpy/ensembles/grand_canonical_ensemble.py`` - ``_commit_or_rollback``
  passes ``self.n_atoms`` to ``_acceptance_condition`` for atomic moves.
- ``mcpy/ensembles/batched_replica_exchange.py`` - ``_batched_single_move``
  calls the same ``_commit_or_rollback`` on each replica.
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

The acceptance has no rotational factor, and orientations are drawn
uniformly.
Uniform orientations do not supply the rotational partition function
:math:`q_\mathrm{rot}` of the reservoir.
Which :math:`\mu` to pass depends on the trial energies:

- **Relaxed trials** (the default): pass the full thermochemical chemical
  potential, :math:`\mu = E_\mathrm{mol} + \Delta\mu(T, p)`, with the
  translational, rotational and internal terms of the gas.
  The relaxation moves each seed into a basin, so the orientation of the
  seed matters little, and the full :math:`\mu` is a good approximation for
  adsorbates.
  It is an approximation, not an exact rule (see
  `Relaxed trials shift the effective mu`_).
- **Single-point trials** (``steps=0`` or a calculator that does not
  relax): pass
  :math:`\mu = E_\mathrm{mol} + k_BT \ln(p\,\Lambda^3 / k_BT)`, with
  :math:`\Lambda` from the total molecular mass and no :math:`q_\mathrm{rot}`.
  This is the convention of LAMMPS ``fix gcmc``.
  The full thermochemical :math:`\mu` would be too low by
  :math:`k_BT \ln q_\mathrm{rot}`.

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
single atoms, so a molecule that relaxation pushes above the cell top stays
both countable and deletable. The floor exclusion is kept on both paths: molecules
that sink below ``bottom_z`` remain excluded, and the buried-species
irreversibility noted above applies to them unchanged.
:class:`mcpy.moves.MoleculeDisplacementMove` picks its candidates with
``cell.is_point_inside(com)`` and rejects a displacement whose new center of
mass fails the same test.
So the region it picks from and the region it moves within agree, and for a
``CustomCell`` the top is a hard wall: no insertion can put a molecule
above it.

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
