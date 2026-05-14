# JOSS Paper: mcpy Case Study System Selection and Paper Plan

## System Recommendation: Pd + Au + H

**Recommended over Ag/Pd/H** for the following reasons:

### Why Pd-Au-H is the best choice

1. **Best-in-class reference data for comparison.** The paper ["A tale of two phase diagrams: Interplay of ordering and hydrogen uptake in Pd-Au-H"](https://www.sciencedirect.com/science/article/pii/S1359645421002731) (Rahm, Lofgren, Fransson et al., *Acta Materialia* 2021) provides full Pd-Au-H phase diagrams constructed with cluster expansions + Monte Carlo. This gives a direct, quantitative comparison target for mcpy results.

2. **Both surface and nanoparticle reference data exist:**
   - Surface adsorption: DFT studies on Pd(111), Pd(100) with ordered H structures (sqrt3 x sqrt3, 1x1), well-characterized adsorption sites
   - Nanoparticles: Multi-plateau hydrogen absorption isotherms for Pd nanoparticles of various shapes (octahedra, cuboctahedra), including shape-dependent effects
   - Bimetallic segregation: In-situ TEM studies of Au-Pd nanoparticle redistribution under H2/O2 atmospheres (J. Mater. Chem. A, 2024)

3. **Showcases all mcpy capabilities:**
   - `InsertionMove` / `DeletionMove` for H adsorption/desorption
   - `PermutationMove` for Au-Pd ordering/segregation (alloy swap moves)
   - `SphericalCell` for nanoparticles, `CustomCell` for slabs
   - `GrandCanonicalEnsemble` and `ReplicaExchange`
   - `analyze_phase_diagram_results` for phase diagram construction

4. **MACE-MH-1 compatibility.** The model (already used in the notebooks) has strong accuracy on adsorption benchmarks (MAE 0.095 eV on S24, 0.138 eV on OC20). Pd, Au, and H are well-represented in the OC20 training data.

5. **Technological relevance.** Pd-Au alloys are actively researched for hydrogen sensing, hydrogen separation membranes, and selective hydrogenation catalysis -- this broadens the paper's appeal.

6. **Existing code scaffold.** The file `examples/gcmc_bimet_nano.py` already implements a Pt-Au+H nanoparticle GCMC simulation. Adapting to Pd-Au+H is straightforward (swap `Pt` for `Pd`, adjust lattice parameters and radii).

### Why not Ag/Pd/H

- Less extensive reference GCMC/MC phase diagram data for direct comparison
- Ag-Pd miscibility gap complicates surface modeling at low temperatures
- Fewer published nanoparticle studies with quantitative phase data

### Alternative worth mentioning: Pt-Cu+H

- Good segregation data under hydrogen (experimental TEM + DFT)
- But lacks the comprehensive phase diagram comparison that Pd-Au-H offers

---

## Case Study Design

### Systems to simulate

- **Pd-Au(111) slab**: 3x3x4, Pd50Au50 random alloy, H adsorbate
- **Pd-Au(100) slab**: 3x3x4, Pd50Au50 random alloy, H adsorbate
- **Pd-Au nanoparticle (small)**: Truncated octahedron ~140 atoms, H adsorbate
- **Pd-Au nanoparticle (large)**: Truncated octahedron ~405 atoms, H adsorbate

### Simulation workflows per system

**Workflow A -- Plain GCMC sweep:**
- Sweep delta_mu_H values (e.g., -1.0 to 0.0 eV, 10-15 points)
- Fixed temperature (e.g., 300 K, 500 K)
- Moves: InsertionMove(H), DeletionMove(H), PermutationMove(Pd, Au)
- Sufficient GCMC steps per point (production: ~5000-10000)

**Workflow B -- RE-GCMC:**
- Temperature ladder (e.g., 300 K to 1200 K, 8-12 replicas)
- Same delta_mu_H sweep
- Demonstrate improved sampling (fewer trapped states, faster convergence)

**Workflow C -- Phase diagram construction:**
- Use `analyze_phase_diagram_results` on both GCMC and RE-GCMC outputs
- Construct T-p(H2) phase diagrams
- Compare: (a) GCMC vs RE-GCMC convergence, (b) mcpy results vs published Pd-Au-H cluster expansion phase diagrams

### Key comparisons to highlight in the paper

- **Sampling quality:** Show that RE-GCMC finds lower free-energy structures and converges faster than plain GCMC (e.g., energy vs MC step, composition fluctuations, acceptance rates)
- **Phase boundaries:** Compare surface phase transitions (ordered H structures, coverage vs delta_mu) against DFT reference data
- **Alloy ordering:** Show how PermutationMove reveals Au surface segregation under hydrogen atmosphere, consistent with experimental observations
- **Size effects:** Nanoparticle multi-plateau isotherms vs single-plateau slab behavior

---

## JOSS Paper Structure

Following [JOSS submission requirements](https://joss.readthedocs.io/en/latest/submitting.html), the `paper.md` should be concise (typically 1000-2000 words) with:

- **Summary**: mcpy is a Python library for (replica-exchange) grand canonical Monte Carlo simulations with machine-learning interatomic potentials, designed for surface adsorption and nanoparticle studies.
- **Statement of need**: Gap between expensive DFT-GCMC (limited sampling) and classical forcefields (limited accuracy); MLIPs bridge this. No existing open-source tool combines RE-GCMC + MLIP + phase diagram analysis in a modular Python framework.
- **Key features**: Modular move system, custom/spherical cells, MACE integration, MPI replica exchange, automated phase diagram analysis.
- **Case study: Pd-Au-H**: Brief description of the system, what was computed, key results, comparison with literature.
- **Figures**: (1) Convergence comparison GCMC vs RE-GCMC, (2) Surface phase diagram, (3) Nanoparticle coverage isotherm.
- **References**: Cite MACE, Pd-Au-H reference papers, RE-GCMC methodology papers.

---

## Implementation Steps

The case study requires writing simulation scripts, running them, and then analyzing the results. Below are the concrete code tasks:

- **Step 1**: Create surface slab scripts for Pd-Au(111) and Pd-Au(100) with GCMC and RE-GCMC configurations
- **Step 2**: Create nanoparticle scripts for truncated octahedron Pd-Au (140 and 405 atoms) with H adsorption
- **Step 3**: Run GCMC sweeps across delta_mu_H values
- **Step 4**: Run RE-GCMC with temperature ladder for the same systems
- **Step 5**: Analyze results with `analyze_phase_diagram_results`, construct phase diagrams
- **Step 6**: Create comparison plots (GCMC vs RE-GCMC, mcpy vs literature)
- **Step 7**: Write `paper.md` following JOSS format

---

## Folder Structure

All JOSS paper materials live under `joss/`:

```
joss/
  plan.md                       -- this plan
  paper.md                      -- JOSS paper (to be written)
  paper.bib                     -- references
  scripts/
    gcmc_surface.py             -- GCMC for Pd-Au slab (111 or 100 via CLI arg)
    regcmc_surface.py           -- RE-GCMC for Pd-Au slab (111 or 100 via CLI arg)
    gcmc_nano.py                -- GCMC for Pd-Au nanoparticle (small/large via CLI arg)
    regcmc_nano.py              -- RE-GCMC for Pd-Au nanoparticle (small/large via CLI arg)
    run_sweep.sh                -- Launcher: sweeps delta_mu_H for all systems
    analyze_results.py          -- Post-processing: coverage, convergence, segregation plots
  results/                      -- Output trajectories and logs (created by scripts)
  figures/                      -- Publication-quality plots (created by analyze_results.py)
```

### Running the simulations

Plain GCMC sweep:
```bash
cd joss/scripts
bash run_sweep.sh /path/to/mace-mh-1.model cuda
```

RE-GCMC sweep (requires 6 MPI ranks for temperature ladder):
```bash
cd joss/scripts
bash run_sweep.sh /path/to/mace-mh-1.model cuda --regcmc
```

Individual runs:
```bash
python gcmc_surface.py 111 -0.5 /path/to/mace-mh-1.model cuda
mpirun -np 6 python regcmc_surface.py 111 -0.5 /path/to/mace-mh-1.model cuda
python gcmc_nano.py small -0.5 /path/to/mace-mh-1.model cuda
mpirun -np 6 python regcmc_nano.py small -0.5 /path/to/mace-mh-1.model cuda
```

Analysis:
```bash
cd joss/scripts
python analyze_results.py --results-dir ../results --figures-dir ../figures
```
