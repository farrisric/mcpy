GCMC on a laptop: Lennard-Jones atoms and dimers
=================================================

Two CPU examples run the grand-canonical loop on a Lennard-Jones (LJ) fluid
in a periodic box.
They need only ASE, and their averages can be checked against LAMMPS
``fix gcmc`` at the same conditions (``benchmark/README.md`` in the
repository).
For a surface with a machine-learned potential, go to
:doc:`gcmc_custom_cell` or :doc:`../getting_started`.


Atomic GCMC: ``examples/00_hello_gcmc.py``
------------------------------------------

The script fills an empty 9 :math:`\sigma` box from a reservoir at
:math:`T^* = 2` and :math:`\mu^* = -3`.
Insertion and deletion change the particle count, and a displacement move
equilibrates the particles already present.
Every random stream is seeded, so the run repeats exactly.

.. literalinclude:: ../../examples/00_hello_gcmc.py
   :language: python

The run takes about three minutes on a laptop CPU.
At the end it prints the average particle count over the second half of the
run next to the LAMMPS value, :math:`\langle N \rangle = 219.2 \pm 1.0`.
One run of 6000 steps lands within a few particles of it.
:doc:`../getting_started` shows the particle-count trace of this exact run.

The run writes two files.
``hello_gcmc.out`` holds one line every 100 steps: the step, the particle
count, the energy, and the acceptance ratio of each move over the last 100
steps.
``hello_gcmc.xyz`` holds the configuration every 100 steps, whether or not
the last move was accepted, and frame 0 is the starting configuration.


Molecular GCMC: ``examples/02_molecule_gcmc.py``
------------------------------------------------

The same box exchanges rigid LJ dimers instead of single atoms.
The chemical potential is that of the whole molecule.
``species`` stays empty because no atomic species is exchanged, and
``molecules`` registers the dimer template under its name.

.. literalinclude:: ../../examples/02_molecule_gcmc.py
   :language: python

The run takes about two minutes on a laptop CPU.
It prints the average molecule count over the second half of the run next
to the LAMMPS value, :math:`\langle N_{\mathrm{mol}} \rangle = 72.5 \pm 1.0`.
The trajectory carries the ``molecule_id`` array, so a restart from it keeps
every molecule whole.
:doc:`../molecular_adsorbates` explains the molecular moves and their
chemical potential.
