Grand Canonical Monte Carlo on an Ag Nanoparticle
=================================================

This tutorial demonstrates a **Grand Canonical Monte Carlo (GCMC)** simulation of oxygen adsorption and desorption on a truncated octahedral Ag nanoparticle.  
The objective is to explore equilibrium coverage as a function of the oxygen chemical potential at a fixed temperature.  
The calculation uses the `GrandCanonicalEnsemble` class from `mcpy`, together with a move selector containing insertion, deletion, and lateral diffusion moves.  

For production studies, the **EMT calculator** used here can be replaced by a **Machine Learning Interatomic Potential (MLIP)** such as MACE, NequIP, or ACE.

Thermodynamic setup
-------------------

Chemical potentials are expressed in eV and correspond to reference values for Ag and O.  
The value of Δμ\ :sub:`O` shifts the oxygen potential relative to the reference, mimicking oxidizing or reducing conditions.  
The system is simulated at constant temperature T = 500 K.

**System preparation**

We first create a truncated octahedral Ag nanoparticle and define a spherical Monte Carlo cell that limits the sampling region for insertions and deletions.  
The spherical cell is constructed with atomic radii that define exclusion zones to prevent overlaps during insertion.

.. code-block:: python

    from ase.cluster import Octahedron
    from mcpy.cell import SphericalCell

    atoms = Octahedron("Ag", 6, 1)
    species_radii = {"Ag": 2.95, "O": 1.4}
    scell = SphericalCell(
        atoms=atoms,
        vacuum=3.0,
        species_radii=species_radii,
        mc_sample_points=100_000,
    )

**Calculator**

To allow fast testing, the EMT calculator is used.  
For realistic energetics, it can be replaced by a trained MACE model, for example:

.. code-block:: python

    from ase.calculators.emt import EMT
    calculator = EMT()
    atoms.calc = calculator

    # Example MACE calculator (replace path with your model)
    # from mcpy.calculators import MACE_F_Calculator
    # calculator = MACE_F_Calculator(
    #     model_paths="/path/to/mace.model",
    #     steps=20,
    #     fmax=0.1,
    #     cueq=False,
    #     device="cpu",
    # )
    # atoms.calc = calculator

**Move selection**

The move selector combines insertion, deletion, and lateral displacement moves for the oxygen adsorbate.  
Probabilities are adjusted to balance adsorption and desorption attempts while promoting local relaxation through lateral diffusion.

.. code-block:: python

    from mcpy.moves import InsertionMove, DeletionMove, LateralDisplacementMove
    from mcpy.moves.move_selector import MoveSelector

    move_selector = MoveSelector(
        [InsertionMove(scell, species=["O"], min_insert=0.6, seed=3675437856), 0.40],
        [DeletionMove(scell, species=["O"], seed=43215423143), 0.40],
        [LateralDisplacementMove(species="O", max_displacement=0.25, seed=13579), 0.20],
    )

**Ensemble definition**

The Grand Canonical ensemble is initialized with temperature, chemical potentials, and the move selector.  
The oxygen chemical potential is modified by Δμ\ :sub:`O` to control the adsorbate coverage.

.. code-block:: python

    from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble
    from mcpy.logging import SimulationLogger

    T = 500.0  # K
    mus = {"Ag": -2.99, "O": -4.91}
    delta_mu_O = -0.50
    mus["O"] += delta_mu_O

    logger = SimulationLogger("gcmc.log", write_every=200)

    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[scell],
        mu=mus,
        species=["O"],
        temperature=T,
        move_selector=move_selector,
        units_type="metal",
        rng_seed=24680,
        logger=logger,
        observables_path="gcmc_observables.csv",
        write_traj_every=2000,
        traj_path="gcmc_trajectory.traj",
    )

**Running the simulation**

The simulation is divided into an equilibration phase followed by production steps.  
After completion, basic metrics such as move acceptance ratios and final coverage are printed.

.. code-block:: python

    gcmc.run(n_steps=50_000, equilibration=5_000)
    atoms.write("gcmc_final.traj")

    acc = gcmc.metrics.get("acceptance", {})
    print("Acceptance ratios:", acc)
    print("Final N(O):", gcmc.observables.get("N_O"))

Interpretation
--------------

After convergence, analyse the resulting observables:

- Plot the oxygen coverage against step number to assess equilibration.
- Repeat the calculation for several values of Δμ\ :sub:`O` to construct a coverage–μ\ :sub:`O` diagram.
- Verify acceptance ratios for each move (ideal range: 10–50 %).  
  Adjust `min_insert`, `max_displacement`, and move probabilities to maintain balance between insertion and deletion.

Common pitfalls
---------------

- Ensure consistent **energy units** between the calculator and ensemble (`units_type='metal'` for eV).  
- Verify that the **spherical cell** has sufficient vacuum to avoid self-interaction.  
- Always perform a short equilibration before collecting statistics.  
- Fix random seeds for full **reproducibility**.  

