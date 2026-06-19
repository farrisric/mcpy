# mcpy

Grand Canonical Monte Carlo sampling of atomistic systems on top of ASE. This glossary fixes the vocabulary for the simulation domain so the same concept is always named the same way. Scope: the GCMC core (ensembles, moves, sampling volume, acceptance). The global-optimization move layer is deliberately out of scope here.

## Language

### Ensembles & sampling

**Ensemble**:
The statistical collection a run samples, defined by which thermodynamic quantities are held fixed; it owns the Monte Carlo loop.
_Avoid_: simulation mode, sampler

**GCMC**:
Sampling at fixed chemical potential, volume, and temperature, where particle number fluctuates through insertions and deletions.
_Avoid_: open-system MC, μVT mode

**Canonical**:
Sampling at fixed particle number, volume, and temperature.
_Avoid_: NVT mode, closed-system MC

**Replica exchange**:
Running independent replicas concurrently and periodically swapping their configurations; the swap runs along either chemical potential or temperature, so it is not strictly parallel tempering.
_Avoid_: parallel tempering (correct only for the temperature axis)

**Replica**:
One independent simulation within a replica-exchange run, pinned to its own chemical potential or temperature.
_Avoid_: walker, worker

### Moves

**Move**:
A proposed change to the configuration that the acceptance criterion then keeps or rejects; the unit of Monte Carlo sampling.
_Avoid_: step, perturbation, mutation

**Insertion / Deletion**:
Moves that add or remove one atom in the sampling volume, changing particle number; their acceptance carries the de Broglie factor.
_Avoid_: spawn/kill, birth/death

**Displacement / Shake / Brownian**:
Moves that perturb the positions of existing atoms without changing particle number.
_Avoid_: jiggle, kick

**Permutation**:
A move that swaps the chemical species of two atoms to sample chemical ordering; particle number is unchanged.
_Avoid_: swap, mutation, exchange (reserve "exchange" for replica exchange)

**Move selector**:
The weighted sampler that chooses which move to attempt each step and tracks per-move acceptance.
_Avoid_: move picker, scheduler

### Sampling volume

**Sampling volume**:
The spatial region within which moves may place or sample atoms. Comes in several shapes (periodic box, sphere around a nanoparticle, dome, custom, null).
_Avoid_: cell (collides with the lattice, below), region, simulation box

**Free volume**:
The portion of the sampling volume actually available for insertion, excluding overlaps with the radii of atoms already present; estimated by Monte Carlo sampling for non-box shapes.
_Avoid_: accessible volume, void volume

**Lattice**:
ASE's periodic cell vectors (`atoms.cell`) that define the simulation box. Named distinctly so "cell" is never reused for the sampling volume.
_Avoid_: cell, unit cell, box (when it risks ambiguity)

### Acceptance & thermodynamics

**Acceptance criterion**:
The Metropolis-style rule deciding whether a trial move is kept, combining the energy change, the inverse temperature, and — for insertion and deletion — the de Broglie factor.
_Avoid_: acceptance condition, accept/reject rule

**Chemical potential (μ)**:
The thermodynamic potential controlling particle exchange with the reservoir; the GCMC control parameter for insertions and deletions.
_Avoid_: mu

**dμ**:
An offset of chemical potential relative to a reference, used to scan conditions and build phase diagrams.
_Avoid_: delta-mu, mu offset

**de Broglie wavelength**:
The thermal wavelength that enters the GCMC acceptance factor for insertion and deletion, setting the entropic cost of changing particle number.
_Avoid_: thermal wavelength

**β**:
Inverse temperature, 1/(k_B·T); the energy scale of the acceptance criterion.
_Avoid_: inverse temp, beta factor

**Species**:
A chemical element treated as an insertable, removable, or permutable identity in the simulation.
_Avoid_: element, atom type, particle type

**Calculator**:
The energy-and-forces backend that returns a configuration's energy to the acceptance criterion.
_Avoid_: potential, model, engine
