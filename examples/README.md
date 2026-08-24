# Examples

Run them in order. The first three need nothing but `pip install -e .` and a CPU.

None of them take command-line flags: every setting is a named constant at the
top of the file. Edit and re-run.

| Example | What it shows | Needs | Runtime |
| --- | --- | --- | --- |
| `00_hello_gcmc.py` | The GCMC loop: cell, insertion/deletion/displacement moves, chemical potential, output files. Lennard-Jones fluid in a box. | ASE only | ~40 s |
| `01_canonical_mc.py` | Canonical (NVT) MC and basin hopping: permutation and displacement moves, LBFGS-relaxed trials, `minima_file`. CuAu nanoparticle with EMT. | ASE only | ~10 s |
| `02_molecule_gcmc.py` | Molecular adsorbates: rigid insertion, deletion and translate+rotate moves, `molecule_id` bookkeeping. Lennard-Jones dimers. | ASE only | ~20 s |
| `03_gcmc_surface_mace.py` | A production run: O and Ag both exchanged on an Ag(111) surface (one cell per species) with a machine-learned potential, FIRE relaxation, and chemical potentials derived from the potential itself. | `nvalchemi-toolkit[mace]`, GPU | hours |
| `04_replica_exchange_batched.py` | Replica exchange on one GPU, with every replica's trial energy evaluated in a single batched forward pass. | `nvalchemi-toolkit[mace]`, GPU | hours |

The first three use toy potentials (Lennard-Jones, EMT) on purpose: they run
anywhere and their reference values are known. Real chemistry needs a real
potential, which is what the last two examples use.

The same material with plots and derivations in between lives in `../notebooks`:
`gcmc_basics_lj.ipynb`, `molecular_gcmc.ipynb`, `grand_canonical_simulation_ag.ipynb`,
`co_on_cupd_replica_exchange.ipynb` and `phase_diagram.ipynb` (which builds a
surface phase diagram from the sweep in `notebooks/phase_diagram_demo`).
The scripts here are the same workflows without the narration, so they can be
run and diffed directly.
