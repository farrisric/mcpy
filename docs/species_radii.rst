Calculating `species_radii`
===========================

This page describes a practical way to choose `species_radii` for GCMC free-volume estimation in `mcpy`.

The exclusion radii defined by `species_radii` determine how Monte Carlo sample points are
classified as occupied vs free, which directly sets the accessible insertion volume
``V_free`` used by GCMC acceptance terms.

Idea
----

Choose radii from relaxed atomistic configurations so that insertion exclusion distances reflect your interaction model.
For each relevant element pair, sample multiple starting geometries, relax, and measure the minimum stable interatomic distance.

Free-volume estimation in the custom insertion cell
-----------------------------------------------------
In `mcpy`, the accessible/free volume used inside Grand Canonical Monte Carlo (GCMC) acceptance criteria
is estimated in a *custom insertion cell* (see `mcpy.cell.CustomCell`).

Monte Carlo sampling draws random points uniformly in the insertion cell. A sampled point is
classified as *occupied* if it lies inside the exclusion sphere of at least one atom. The
exclusion spheres are defined by `species_radii` (an element-wise mapping from chemical
species to an exclusion radius).

Let :math:`V_cell` be the cell volume and :math:`N_MC` the number of Monte Carlo sample points. For each random point :math:`\mathbf{x}_k`, define an indicator

.. math::

   I_k =
   \begin{cases}
   1, & \exists\, a\ \text{such that}\ \|\mathbf{x}_k-\mathbf{r}_a\|^2 \le r_{\mathrm{species}(a)}^2,\\
   0, & \text{otherwise.}
   \end{cases}

Then the occupied fraction and free volume are

.. math::

   f_{\mathrm{occ}} = \frac{1}{N_{\mathrm{MC}}}\sum_{k=1}^{N_{\mathrm{MC}}} I_k,
   \qquad
   V_{\mathrm{free}} = V_{\mathrm{cell}}\,(1 - f_{\mathrm{occ}}).

Finally, this :math:`V_free` is used in the place of the geometric :math:`V` when computing the GCMC insertion/deletion acceptance probabilities (see the equations in :doc:`examples`).

Insertion region geometry (how the “cell” is chosen)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the thesis, the insertion region is selected to restrict sampling to chemically relevant spatial regions:

Surface slabs: the insertion region is a sub-volume that spans the full :math:`x` and :math:`y` dimensions of the simulation cell and a finite thickness along :math:`z` (e.g. a top slab subregion of height :math:`\sim 5.5\ \AA`).

Nanoparticles: each nanoparticle is centered in a spherical insertion region chosen to enforce a vacuum gap (e.g. :math:`\sim 3\ \AA`) between the outermost particle atom and the insertion boundary.

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

Using `utils/compute_radii.py`
------------------------------

`mcpy` already includes a ready-to-use script at `utils/compute_radii.py` that automates this workflow:

- builds an FCC(111) metal slab,
- inserts trial atoms many times in a custom insertion cell,
- relaxes each trial structure with your MACE model,
- stores insertion and relaxed nearest-neighbor distances in `*.npy`,
- and writes a histogram/KDE figure (`dist_hist.png`) to identify representative relaxed distances.

Inputs / Outputs
~~~~~~~~~~~~~~~~~
Input:

- A path to your trained MACE model (passed as the single command-line argument).

Outputs (for each inserted species):

- `<species>_distances.npy`: pairs of ``(d_insertion, d_relaxed)`` nearest-neighbor distances.
- `dist_hist.png`: histogram/KDE plot used to pick a conservative exclusion distance.
- `insertion.log`: logging of insertion/relaxation progress.

Configure the script
~~~~~~~~~~~~~~~~~~~~

In `utils/compute_radii.py`, set the key parameters:

- `metal_species` (for example `Ag`),
- `gas_species` (for example `O`),
- `lattice_param`,
- `cell_bottom` and `cell_height`,
- `n_trials`,
- and relaxation settings (`relax_max_steps`, `relax_fmax`).

Run the script
~~~~~~~~~~~~~~

.. code-block:: bash

   python utils/compute_radii.py /path/to/your_mace_model.model

Interpretation for `species_radii`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For O on Ag, inspect the relaxed O-Ag distance distribution (`O_distances.npy` and `dist_hist.png`) and select a conservative minimum stable O-Ag distance from your relaxed trials.

Then define `species_radii` with your chosen convention, for example:

.. code-block:: python

   species_radii = {"Ag": d_min_O_Ag, "O": 0.0}

This keeps the same physical calibration method used by the script while matching the element-wise format expected by `mcpy`.

