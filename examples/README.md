# Examples

Run them in order. The first three need nothing but `pip install -e .` and a CPU.

None of them take command-line flags: every setting is a named constant at the
top of the file. Edit and re-run.

| Example | What it shows | Needs | Runtime |
| --- | --- | --- | --- |
| `00_hello_gcmc.py` | The GCMC loop: cell, insertion/deletion/displacement moves, chemical potential, output files. Lennard-Jones fluid in a box; prints <N> next to the LAMMPS `fix gcmc` value. | ASE only | ~3 min |
| `01_canonical_mc.py` | Canonical (NVT) MC and basin hopping: single and compound permutation moves, LBFGS-relaxed trials, `minima_file`. Au segregates to the surface of a CuAu nanoparticle (EMT). | ASE only | ~10 s |
| `02_molecule_gcmc.py` | Molecular adsorbates: rigid insertion, deletion and translate+rotate moves, `molecule_id` bookkeeping. Lennard-Jones dimers; prints <N_mol> next to the LAMMPS `fix gcmc mol` value. | ASE only | ~2 min |
| `03_gcmc_surface_mace.py` | O and Ag both exchanged on an Ag(111) surface (one cell per species) with the small AGNESI MACE potential and LBFGS-relaxed trials, with the settings of the Ag oxidation study. Runs on a CPU; the model downloads on the first run. | `pip install -e .[mace]` | hours |
| `04_replica_exchange_batched.py` | Replica exchange on one GPU, with every replica's trial energy evaluated in a single batched forward pass. | `pip install -e .[alchemi]`, CUDA GPU | hours |
| `05_pt_nanoparticle_h_che.py` | H on a Pt nanoparticle versus electrode potential: a replica-exchange ladder over U_SHE, mu_H from the computational hydrogen electrode, MACE mh-1 with cuEq and LBFGS relaxation. | `pip install -e .[alchemi]`, nvalchemi from GitHub main, CUDA GPU | hours |

The first three use toy potentials (Lennard-Jones, EMT) on purpose: they run
anywhere and their reference values are known. Real chemistry needs a real
potential, which is what the last two examples use.

The same material with plots and derivations in between lives in `../notebooks`:
`gcmc_basics_lj.ipynb`, `molecular_gcmc.ipynb`, `grand_canonical_simulation_ag.ipynb`,
`co_on_cupd_replica_exchange.ipynb` and `phase_diagram.ipynb`.
The documentation shows every script in full: <https://mc-py.readthedocs.io/en/latest/examples.html>.
The scripts here are the same workflows without the narration, so they can be
run and diffed directly.
