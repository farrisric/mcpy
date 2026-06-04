Computing energies and relaxing trial moves
============================================

Intent
------

For each trial configuration the Monte Carlo loop asks the calculator for an
energy. In `mcpy` the answer usually includes a short local relaxation first. A
calculator wrapper takes the trial structure, runs a capped geometry
optimization, and returns the energy of the relaxed configuration.

This coupling is what lets GCMC accept insertions and deletions in dense
systems. A raw insertion drops an atom at a random point in the free volume,
often a fraction of an angstrom from its eventual bonding site. Evaluated as
given, that structure carries a large repulsive energy and the move is almost
always rejected. The hybrid scheme of Senftle et al. relaxes the trial first, so
the energy fed to the acceptance rule reflects the nearby local minimum rather
than the random starting point. Acceptance then tracks real chemistry, and the
run reaches stable behaviour in roughly an order of magnitude fewer steps.

Read this page when choosing a backend, deciding how much relaxation to spend
per trial, or scaling a run onto a GPU.


Design decisions
----------------

**Relaxation runs inside energy evaluation.** Every wrapper exposes one method,
``get_potential_energy(atoms)``, and performs the relaxation there. The energy
handed to the Metropolis or de Broglie acceptance rule is therefore always the
relaxed energy. Keeping the two steps together removes any risk of comparing a
relaxed energy against an unrelaxed one, which would corrupt detailed balance.

**Any ASE calculator works through** ``BaseCalculator``. `mcpy` does not require
a bespoke energy backend. ``BaseCalculator`` wraps a configured ASE calculator
and drives an LBFGS relaxation around it, so a DFT code, a classical potential,
or an MLIP such as NequIP or ACE plugs in without new code. The MACE and Alchemi
classes are conveniences over this same pattern.

**Caps bound the relaxation, they do not converge it.** Each trial relaxes for
at most ``steps`` optimizer iterations or until the maximum force drops below
``fmax``. Short caps are deliberate. A Monte Carlo run spends its budget on many
trials, not on converging each one to machine precision. A cap set too low
truncates the relaxation before the energy settles and biases the energies that
enter acceptance.

**MACE has a single-point class and a relaxing class.** ``MACECalculator``
returns a bare forward pass with no optimization. ``MACE_F_Calculator`` relaxes
first and is the workhorse for hybrid GCMC. Keeping them separate leaves the
no-relaxation workflow available for systems where insertions land in open space
and need no local optimization.

**The Alchemi backend keeps relaxation on the GPU.** ASE optimizers move
positions and forces between the model and the CPU on every step. Past roughly a
hundred atoms that round trip dominates the wall time. The optional
``AlchemiCalculator`` and ``AlchemiFCalculator`` run the model and a
GPU-resident FIRE optimizer in one device context, which is what makes
thousand-atom GCMC tractable. The backend is an optional import, so a missing
``nvalchemi`` install leaves the rest of the library working.

**Fixed atoms stay fixed during relaxation.** Substrate atoms carrying an ASE
``FixAtoms`` constraint must not drift while the adsorbate relaxes. The ASE
optimizers honour the constraint natively. The Alchemi FIRE optimizer has no
notion of ``FixAtoms``, so `mcpy` tags those atoms, zeroes their forces on every
step, and restores their exact positions on write-back. Without this the
returned energy would describe a geometry the run never keeps.

**Accepted trials keep their relaxed positions.** After relaxation the wrapper
writes the optimized coordinates back into the ``Atoms`` object, skipping fixed
indices. An accepted move advances the configuration to the relaxed structure
rather than the random proposal, so the next trial builds on a physical
geometry.

**Batched evaluation drives single-GPU replica exchange.**
``get_potential_energies(atoms_list)`` evaluates a list of structures in one set
of kernel launches instead of one launch per structure.
``BatchedReplicaExchange`` uses this to step every replica through a single
forward pass per move, faster on one GPU than launching a separate context per
replica.


Driving any ASE calculator
--------------------------

``BaseCalculator(calculator, steps, fmax)``
   The general adapter. Give it a constructed ASE calculator, a maximum number
   of LBFGS steps, and a force tolerance in eV/Å. Use it for any potential
   without a dedicated wrapper, including DFT codes and MLIPs other than MACE.

``BaseCalculator.get_potential_energy(atoms)``
   Attaches the calculator to ``atoms``, runs LBFGS up to ``steps`` or ``fmax``,
   and returns the relaxed energy. The ``atoms`` object is relaxed in place.


Running MACE potentials
-----------------------

``MACECalculator(model_paths, device='cpu')``
   A single forward pass of a MACE model with no relaxation. Loads the model
   once, freezes its parameters, and returns the energy of the structure as
   given. Use it when the move set needs only ``E(atoms)``, for example a
   displacement-only run on already-relaxed geometries.

``MACE_F_Calculator(model_paths, steps, fmax, device='cpu', cueq=False, optimizer='lbfgs')``
   The relax-then-energy MACE wrapper used in most GCMC runs. Pass a model path
   or a pre-built ``MACECalculator`` to reuse. ``optimizer`` selects ``'lbfgs'``
   or ``'fire'``. Choose ``'fire'`` when comparing against the Alchemi backend,
   which also uses FIRE. The wrapper records ``last_relax_steps`` after each
   call.


Scaling to large systems on GPU
-------------------------------

The Alchemi classes are an optional backend. Install them with
``pip install -e .[alchemi]``, which pulls in ``nvalchemi-toolkit[mace]`` and
needs a CUDA-enabled PyTorch build. Set ``compile_model=False`` for GCMC, where
the atom count changes between trials and a compiled static graph does not hold.

``AlchemiCalculator(checkpoint='medium-mpa-0', device='cuda', enable_cueq=True, compile_model=True)``
   GPU-native MACE evaluation with no relaxation. Accepts a named checkpoint, a
   local ``.pt`` path, or a shared ``MACEWrapper``. ``enable_cueq`` turns on
   fused equivariance kernels.

``AlchemiFCalculator(checkpoint='medium-mpa-0', steps=500, fmax=0.05, dt=1.0, optimizer='fire')``
   The GPU counterpart of ``MACE_F_Calculator``. Relaxes with a GPU-resident
   FIRE optimizer, honouring ``FixAtoms``, then returns the relaxed energy.
   ``dt`` is the FIRE initial timestep. The default of 1.0 converges in about
   half the steps of a smaller value. ``optimizer`` selects ``'fire'`` or the
   ``'fire2'`` variant.

``get_potential_energies(atoms_list)``
   Evaluates or relaxes a list of structures of possibly different sizes in one
   batched pass, returning one energy per structure. The relaxing form retires
   each structure from the active batch as its force converges, so finished
   replicas do not slow the rest. This is the hook ``BatchedReplicaExchange``
   calls each step.

``run_md(atoms, temperature, friction=0.01, dt=2.0, steps=100, seed=42)``
   Runs NVT Langevin dynamics in place, reusing the loaded model and neighbor
   list. It backs the GPU Brownian move rather than the GCMC acceptance loop,
   and appears here because it shares the calculator's model.


Tuning the relaxation
---------------------

``steps`` / ``fmax``
   The relaxation budget. Raise ``steps`` or lower ``fmax`` for tighter minima
   at a higher cost per trial. Setting ``steps`` too low truncates the
   relaxation and biases the energies that enter acceptance.

``optimizer``
   LBFGS or FIRE on the ASE path, ``'fire'`` or ``'fire2'`` on the Alchemi path.
   Match the optimizer across backends when comparing energies or step counts.

``FixAtoms``
   Apply an ASE ``FixAtoms`` constraint to hold substrate atoms during
   relaxation. Every backend keeps the constrained indices at their original
   positions.

``last_relax_steps`` / ``total_relax_steps``
   Two counters updated after each evaluation. ``last_relax_steps`` holds the
   steps the last relaxation used. ``total_relax_steps`` holds the running
   total. Read them to confirm relaxations converge inside the ``steps`` cap
   rather than hitting it every time.
