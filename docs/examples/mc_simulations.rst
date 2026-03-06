Canonical Monte Carlo (constant NVT) on a small Ag nanoparticle.
================================================================

Replace the EMT calculator with your MLIP (e.g., MACE) for production.


.. code-block:: python

    from __future__ import annotations


    import numpy as np
    from ase.build import fcc111
    from ase.calculators.emt import EMT
    from ase import Atoms


    # --- mcpy imports ---
    from mcpy import CanonicalEnsemble, MoveSelector
    from mcpy.moves import TranslationMove, SwapMove
    from mcpy.logging import SimulationLogger


    # System: a small Ag slab fragment as a proxy nanoparticle
    slab = fcc111('Ag', size=(3, 3, 3), vacuum=8.0) # small demo system
    slab.calc = EMT() # swap with MACE for research: MACECalculator(model_path=...)


    # Ensemble parameters
    T = 500.0 # K
    n_steps = 20_000


    # Moves
    moves = MoveSelector([
    (TranslationMove(max_displacement=0.15), 0.85),
    (SwapMove(symbol_a='Ag', symbol_b='Ag'), 0.15), # placeholder; for alloys, set real species
    ])


    # Logger
    logger = SimulationLogger('outputs/mc.log', write_every=100)


    # Canonical MC
    ensemble = CanonicalEnsemble(
    atoms=slab,
    temperature=T,
    moves=moves,
    rng_seed=42,
    logger=logger,
    )


    # Run
    ensemble.run(n_steps=n_steps, equilibration=2_000)


    # Save final structure
    slab.write('outputs/mc_final.traj')
    print('MC completed. Final energy (eV):', slab.get_potential_energy())
