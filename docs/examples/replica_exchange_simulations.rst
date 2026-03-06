Canonical Monte Carlo (constant NVT) on a small Ag nanoparticle.
================================================================

Replace the EMT calculator with your MLIP (e.g., MACE) for production.


.. code-block:: python
    
    """
    Replica-Exchange GCMC (RE-GCMC) swapping across chemical potentials of O.
    Replicas differ by mu_O; periodic exchange attempts enhance sampling.
    """
    from __future__ import annotations


    import numpy as np
    from ase.build import fcc111
    from ase.calculators.emt import EMT


    # --- mcpy imports ---
    from mcpy import GrandCanonicalEnsemble, ReplicaExchange, MoveSelector
    from mcpy.moves import AdsorbMove, DesorbMove, LateralDisplacementMove
    from mcpy.chemistry import ChemicalPotential
    from mcpy.logging import SimulationLogger


    # Common structure
    base = fcc111('Ag', size=(4, 4, 4), vacuum=10.0)
    base.pbc = (True, True, False)
    base.calc = EMT() # replace with MLIP


    T = 600.0 # K


    # Ladder of chemical potentials (eV)
    mu_values = [-0.6, -0.8, -1.0, -1.2, -1.4]


    replicas = []
    for i, mu in enumerate(mu_values):
    atoms_i = base.copy()
    moves = MoveSelector([
    (AdsorbMove(species='O', candidate_sites='fcc', max_height=1.8), 0.40),
    (DesorbMove(species='O'), 0.40),
    (LateralDisplacementMove(species='O', max_displacement=0.20), 0.20),
    ])
    logger = SimulationLogger(f'outputs/rex_rep{i}.log', write_every=200)
    ensemble = GrandCanonicalEnsemble(
    atoms=atoms_i,
    temperature=T,
    chemical_potentials=[ChemicalPotential('O', mu, 'eV')],
    moves=moves,
    rng_seed=100 + i,
    logger=logger,
    )
    replicas.append(ensemble)


    rex = ReplicaExchange(
    replicas=replicas,
    swap_interval=1000, # attempt swaps every 1000 MC steps
    scheme='neighbor', # swap only adjacent chemical potentials
    rng_seed=2025,
    )


    # Run 10k steps per replica with exchanges
    rex.run(n_steps=10_000, equilibration=2_000)


    # Collect and save data
    rex.save_observables('outputs/rex_observables.csv')
    for i, ens in enumerate(rex.replicas):
    ens.atoms.write(f'outputs/rex_final_rep{i}.traj')


    print('RE-GCMC completed. Swap acceptance:', rex.metrics.get('swap_acceptance'))
