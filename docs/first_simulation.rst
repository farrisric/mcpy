Your first simulation
=====================

This page walks through a complete, runnable GCMC simulation: a small Ag(111)
slab exposed to an oxygen reservoir at fixed temperature and chemical
potential. It is the shortest path from a fresh install to a trajectory and
log on disk.

If you have not installed `mcpy` yet, start with :doc:`installation`.


Goal
----

Sample the equilibrium oxygen coverage on a 4×4×3 Ag(111) slab at
:math:`T = 500\,\mathrm{K}` and :math:`\mu_{\mathrm{O}} = -5.0\,\mathrm{eV}`,
using a MACE potential for energy evaluation.

After the run you should have:

- ``gcmc_first.xyz`` — the extended-XYZ trajectory,
- ``gcmc_first.out`` — the step-by-step log with per-move acceptance ratios.


Inputs
------

============================  ===========================================
Setting                       Value
============================  ===========================================
Surface                       fcc(111), 4×4×3 Ag slab, 8 Å vacuum
Adsorbate                     O (single species)
Temperature                   500 K
:math:`\mu_{\mathrm{O}}`      :math:`-5.0\,\mathrm{eV}`
Cell                          ``CustomCell`` 5 Å above the top Ag layer
Moves                         ``InsertionMove`` + ``DeletionMove`` for O
Calculator                    MACE (e.g. ``mace_mp`` medium model)
Steps                         200 GCMC steps
Random seed                   42
============================  ===========================================

The :doc:`species_radii` page explains how to calibrate the exclusion radii;
for this first run, sane defaults are used.


Code
----

.. code-block:: python

   import numpy as np
   from ase.build import fcc111
   from ase.constraints import FixAtoms

   from mcpy.calculators import MACE_F_Calculator
   from mcpy.cell import CustomCell
   from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
   from mcpy.moves import InsertionMove, DeletionMove
   from mcpy.moves.move_selector import MoveSelector

   # 1. Build the substrate and fix the bottom layer.
   atoms = fcc111('Ag', a=4.085, size=(4, 4, 3), vacuum=8.0, periodic=True)
   bottom = [a.index for a in atoms if a.tag == 3]
   atoms.set_constraint(FixAtoms(indices=bottom))

   # 2. Define the insertion region: a thin slab 5 Å above the surface.
   cell = CustomCell(
       atoms,
       custom_height=5.0,
       bottom_z=atoms.positions[:, 2].max() + 0.5,
       species_radii={'Ag': 2.75, 'O': 0.0},
   )

   # 3. Wire up the calculator (use your local MACE checkpoint).
   calculator = MACE_F_Calculator(
       model_paths='mace_mp_medium.model',
       steps=40,
       fmax=0.1,
       device='cuda',
   )

   # 4. Move set: equal weight on insertion and deletion of O.
   ss = np.random.SeedSequence(42)
   s1, s2 = (int(x) for x in ss.generate_state(2, dtype=np.uint32))
   moves = MoveSelector(
       [1, 1],
       [InsertionMove(cell, species=['O'], min_insert=0.5, seed=s1),
        DeletionMove(cell, species=['O'], seed=s2)],
   )

   # 5. Assemble and run.
   gcmc = GrandCanonicalEnsemble(
       atoms=atoms,
       cells=[cell],
       calculator=calculator,
       mu={'O': -5.0},
       units_type='metal',
       species=['O'],
       temperature=500.0,
       move_selector=moves,
       outfile='gcmc_first.out',
       traj_file='gcmc_first.xyz',
       outfile_write_interval=1,
       trajectory_write_interval=1,
   )
   gcmc.run(steps=200)


Run it
------

.. code-block:: bash

   python first_simulation.py

The first step is slower than the rest — the calculator is being initialised
and the cell free volume is sampled. On a single GPU, 200 steps complete in
a few minutes for this system size.


Expected output
---------------

The log file ``gcmc_first.out`` follows the schema documented in
:doc:`ensembles`. Each line carries the step index, particle count, current
energy, and per-move acceptance ratios since the last log line:

.. code-block:: text

   step       N           energy           acceptance
   ----------------------------------------------------------------
   0          48          -157.21          (initial)
   10         51          -160.83          ins: 40.0%, del: 0.0%
   20         53          -163.04          ins: 30.0%, del: 10.0%
   ...
   200        58          -168.12          ins: 25.5%, del: 12.0%

The trajectory ``gcmc_first.xyz`` can be opened directly in ASE:

.. code-block:: python

   from ase.io import read
   traj = read('gcmc_first.xyz', index=':')
   print(f'{len(traj)} frames, final N = {len(traj[-1])}')


Interpretation
--------------

- The oxygen count :math:`N` should climb from zero and stabilise around the
  equilibrium coverage for the chosen :math:`\mu_{\mathrm{O}}`. Plot
  :math:`N` against step to confirm convergence.
- Insertion and deletion acceptance ratios should be comparable in the
  steady state. A persistent imbalance means the system has not yet
  equilibrated, or the exclusion radii are mis-calibrated.
- The final energy is *not* directly comparable across runs at different
  :math:`\mu`; build a phase diagram (see
  :doc:`tutorials/oxidation_phase_diagram`) to compare conditions.


Next steps
----------

- Calibrate ``species_radii`` for your own system: :doc:`species_radii`.
- Add displacement moves for a less localised search: :doc:`moves`.
- Choose a cell that matches your geometry (slab vs. nanoparticle):
  :doc:`cells`.
- Sweep :math:`\mu_{\mathrm{O}}` to build a phase diagram:
  :doc:`tutorials/oxidation_phase_diagram`.
- Run several conditions in parallel with Replica Exchange:
  :doc:`ensembles` (section ``ReplicaExchange``).
