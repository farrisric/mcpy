Canonical Monte Carlo: chemical ordering in a CuAu nanoparticle
===============================================================

``examples/01_canonical_mc.py`` uses :class:`~mcpy.ensembles.CanonicalEnsemble`
to sample the chemical ordering of a 55-atom Cu\ :sub:`28`\ Au\ :sub:`27`
icosahedron at 500 K with ASE's EMT potential.
The atom count never changes.
Two permutation moves explore the ordering instead: a single Cu-Au swap, and
a compound move that swaps four pairs in one trial.
Each trial is relaxed with LBFGS before the Metropolis test, so the run is
also a basin-hopping search, and ``minima_file`` keeps every configuration
that beats the previous best.

.. literalinclude:: ../../examples/01_canonical_mc.py
   :language: python

The run takes about ten seconds on a laptop CPU.
The particle starts from a random mix.
At the end every Au atom sits on the surface (the script prints 27 of 42
surface sites), because Au is the larger atom with the lower surface energy.

Read the acceptance ratios in the final log line.
The single swap is accepted about half the time.
The four-pair jump is accepted less than one time in ten, because four
random swaps rarely all lower the energy, but each accepted jump moves the
ordering much further.

The run writes three files.
``canonical.out`` holds the step and the current energy every 10 steps.
``canonical.xyz`` holds the configuration every 10 steps, whether or not the
last move was accepted.
``canonical_minima.xyz`` holds every new lowest-energy configuration, each
tagged with its step.

For variable composition, go to :doc:`gcmc_simulations`.
For the full move set, go to :doc:`../moves`.
