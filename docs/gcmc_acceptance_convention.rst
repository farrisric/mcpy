GCMC acceptance: the de Broglie particle count
==============================================

This note records a deliberate, non-obvious convention in the grand-canonical
acceptance test, the discussion behind it, and the change history. It exists so
the choice is not silently "fixed" by a future contributor who (correctly)
notices it departs from the textbook/LAMMPS form.

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

per-species exchangeable count (textbook / LAMMPS)
    ``N`` = number of atoms *of the exchanged species* inside the insertion
    region (the move's cell). This is the physically standard choice.

The physically correct quantity is the **per-species** count. The de Broglie
factor is the ideal-gas combinatorial term: the :math:`1/N!` indistinguishability
correction applies only among identical, interchangeable particles, and only to
the population that shares the insertion volume ``V``. The total atom count
mixes in distinguishable species and the fixed substrate, which do not belong in
that factor. The interaction with all those atoms enters the acceptance through
:math:`\Delta E` (the total potential energy change), not through ``N``.


What LAMMPS does
----------------

``fix gcmc`` (``src/MC/fix_gcmc.cpp``) uses exactly the per-species, in-region
count. Its acceptance lines are::

    // insertion
    random < zz*volume*exp(-beta*insertion_energy)/(ngas+1)
    // deletion
    random < ngas*exp(beta*deletion_energy)/(zz*volume)
    // activity
    zz = exp(beta*chemical_potential)/pow(lambda,3.0);

so :math:`p_\text{ins} = V/((n_\text{gas}+1)\Lambda^3)\,e^{-\beta(\Delta U-\mu)}`,
identical in form to mcpy. ``ngas`` is built by ``update_gas_atoms_list()``,
which keeps only atoms passing both the group/type filter (``mask[i] & groupbit``,
restricted to the exchanged type) and the region test (``region->match(...)``).
LAMMPS also confirms two further conventions mcpy shares or should note:

- ``mu`` is the **full** chemical potential, including the ideal/:math:`\Lambda`
  term (``zz = exp(beta*mu)/lambda^3``). So ``mu`` must be referenced on the
  calculator's absolute energy scale (e.g. an O\ :sub:`2` gas reference at the
  target ``T`` and ``p``).
- ``volume`` is the **geometric** region volume (MC-sampled only to handle odd
  region shapes); it does **not** exclude space occupied by atoms. mcpy instead
  uses a *free* volume (``cell_volume * (1 - occupied_fraction)``) while still
  sampling insertion points over the full cell box — an internal inconsistency
  LAMMPS does not have.

Sources: https://docs.lammps.org/fix_gcmc.html and
https://github.com/lammps/lammps/blob/develop/src/MC/fix_gcmc.cpp


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
