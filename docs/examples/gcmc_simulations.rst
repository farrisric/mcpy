Canonical Monte Carlo (constant NVT) on a small Ag nanoparticle.
================================================================

Replace the EMT calculator with your MLIP (e.g., MACE) for production.


.. code-block:: python
    
    from __future__ import annotations


    import numpy as np
    from ase.build import fcc111
    from ase.calculators.emt import EMT


    # --- mcpy imports ---
    from mcpy import GrandCanonicalEnsemble, MoveSelector
    from mcpy.moves import AdsorbMove, DesorbMove, LateralDisplacementMove
    from mcpy.chemistry import ChemicalPotential
    from mcpy.logging import SimulationLogger


    # Build substrate
    slab = fcc111('Ag', size=(4, 4, 4), vacuum=10.0)
    slab.pbc = (True, True, False)
    slab.calc = EMT() # swap with MLIP for production


    # Thermodynamic state
    T = 600.0 # K
    mu_O = ChemicalPotential(species='O', value=-1.10, reference='eV') # example value


    # Move set: adsorption, desorption, lateral diffusion
    moves = MoveSelector([
    (AdsorbMove(species='O', candidate_sites='fcc', max_height=1.8), 0.40),
    (DesorbMove(species='O'), 0.40),
    (LateralDisplacementMove(species='O', max_displacement=0.20), 0.20),
    ])


    logger = SimulationLogger('outputs/gcmc.log', write_every=100)


    ensemble = GrandCanonicalEnsemble(
    atoms=slab,
    temperature=T,
    chemical_potentials=[mu_O],
    moves=moves,
    rng_seed=7,
    logger=logger,
    )


    # Production parameters
    n_steps = 50_000
    ensemble.run(n_steps=n_steps, equilibration=5_000)


    # Dump observables
    ensemble.save_observables('outputs/gcmc_observables.csv')
    slab.write('outputs/gcmc_final.traj')
    print('GCMC completed. Coverage (theta):', ensemble.observables.get('coverage_O', None))
