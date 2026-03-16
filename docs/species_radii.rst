Calculating `species_radii`
===========================

This page describes a practical way to choose `species_radii` for GCMC free-volume estimation in `mcpy`.

Idea
----

Choose radii from relaxed atomistic configurations so that insertion exclusion distances reflect your interaction model.
For each relevant element pair, sample multiple starting geometries, relax, and measure the minimum stable interatomic distance.

Example: O on Ag surface
------------------------

For oxygen insertion on silver surfaces, one practical workflow is:

1. Build representative Ag surface slabs (facets and coverages relevant to your simulation).
2. Place an O atom in many initial positions (top, bridge, hollow, and random lateral positions; different heights).
3. Relax each structure with the same calculator/settings used in production.
4. For each relaxed structure, compute nearest-neighbor distances between O and Ag atoms.
5. Collect all O-Ag nearest distances and take the minimum physically stable value.

That minimum O-Ag distance can be used to define the O/Ag exclusion scale in `species_radii`.

How to map pair distances to `species_radii`
--------------------------------------------

`species_radii` is element-wise, while your measurements are pair-wise distances.
Use a consistent convention across your simulations.

Common choices are:

- Put the full pair distance on the host species and zero on the inserted species.
  Example for O insertion in Ag: `{'Ag': d_min(O-Ag), 'O': 0.0}`.
- Split a pair distance between species.
  Example: `{'Ag': 0.5 * d_min(O-Ag), 'O': 0.5 * d_min(O-Ag)}`.

In all cases, validate acceptance behavior and free-volume stability on short pilot runs before long production sampling.

Reference implementation sketch
-------------------------------

.. code-block:: python

   import numpy as np

   def min_pair_distance(atoms, a_symbol, b_symbol):
       idx_a = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == a_symbol]
       idx_b = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == b_symbol]
       if not idx_a or not idx_b:
           raise ValueError(f"Missing species: {a_symbol} or {b_symbol}")

       dmin = np.inf
       for i in idx_a:
           for j in idx_b:
               if i == j:
                   continue
               d = atoms.get_distance(i, j, mic=True)
               if d < dmin:
                   dmin = d
       return float(dmin)


   # After generating and relaxing many O-on-Ag configurations:
   # d_values = [min_pair_distance(relaxed_atoms_k, "O", "Ag") for k in configs]
   # d_omin_ag = min(d_values)
   # species_radii = {"Ag": d_omin_ag, "O": 0.0}

