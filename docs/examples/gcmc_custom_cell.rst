Grand Canonical Monte Carlo on a small Ag(111) surface with a custom simulation cell
===================================================================================

This tutorial demonstrates **Grand Canonical Monte Carlo (GCMC)** on an Ag(111) slab where
adsorbate insertion/deletion is restricted to a prescribed volume above the surface via a
``CustomCell``. Two distinct cells are used: one targeting Ag atoms and another targeting O atoms.
Chemical potentials are referenced to bulk Ag and ½O₂, then shifted by Δμ\ :sub:`O` to explore
oxidizing conditions at fixed temperature.

Concept of the custom cell
--------------------------
A custom simulation cell restricts trial insertions and deletions to a region directly above the
surface, improving efficiency and avoiding spurious attempts deep in the slab. The lower boundary
(``bottom_z``) anchors the cell a small distance above the topmost Ag layer; ``custom_height`` defines
the accessible vertical extent. Different effective radii can be used in each cell to encode distinct
overlap/clearance rules for Ag versus O manipulations.

Relaxation-style GCMC acceptance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In dense metal systems, direct trial insertions can have low acceptance because the surrounding local
geometry is initially unfavorable. A common remedy is to evaluate the acceptance criterion using a
shortly relaxed version of the trial configuration (energy relaxation step; Senftle et al., 2013).
This strategy is especially relevant when insertion/deletion is restricted to a physically constrained
region above the surface (Wexler et al., 2019).

System construction
-------------------
We build a periodic Ag(111) slab, fix the bottom layer, and define two ``CustomCell`` instances
for Ag and O, respectively. The values for ``bottom_z`` and ``custom_height`` should be adapted
to your lattice constant and vacuum spacing.

.. code-block:: python

    from ase.build import fcc111
    from ase.constraints import FixAtoms
    from mcpy.cell import CustomCell

    atoms = fcc111('Ag', a=4.165, size=(4, 4, 3), periodic=True, vacuum=8.0)
    bottom_layer = [a.index for a in atoms if a.tag == 3]
    atoms.set_constraint(FixAtoms(indices=bottom_layer))

    cell_ag_ag = CustomCell(
        atoms=atoms,
        custom_height=7.0,
        bottom_z=12.8 - 2.068,
        species_radii={'Ag': 2.947, 'O': 0.0},
    )

    cell_ag_o = CustomCell(
        atoms=atoms,
        custom_height=7.0,
        bottom_z=12.8 - 2.068,
        species_radii={'Ag': 2.068, 'O': 0.0},
    )

Calculator
----------
For a quick functional test, you may start with EMT. For production, switch to your
trained MACE model. Keep units consistent with the ensemble (energies in eV).

**EMT option**

.. code-block:: python

    from ase.calculators.emt import EMT
    calculator = EMT()
    atoms.calc = calculator

**MACE option**

.. code-block:: python

    from mcpy.calculators import MACE_F_Calculator
    calculator = MACE_F_Calculator(
        model_paths='/path/to/mace-small-density-agnesi-stress.model',
        steps=20,
        fmax=0.1,
        cueq=False,
        device='cpu',
    )
    atoms.calc = calculator

Move selection
--------------
We combine four moves: deletion and insertion for Ag within the Ag-targeting cell, and
deletion and insertion for O within the O-targeting cell. Probabilities are chosen to balance
the four operations; ``min_insert`` controls the minimal clearance on insertion.

.. code-block:: python

    from mcpy.moves import DeletionMove, InsertionMove
    from mcpy.moves.move_selector import MoveSelector

    move_list = [
        [25, 25, 25, 25],
        [
            DeletionMove(cell_ag_ag, species=['Ag'], seed=12346783764),
            DeletionMove(cell_ag_o,  species=['O'],  seed=43215423143),
            InsertionMove(cell_ag_ag, species=['Ag'], min_insert=0.5, seed=6758763657),
            InsertionMove(cell_ag_o,  species=['O'],  min_insert=0.5, seed=3675437856),
        ],
    ]
    move_selector = MoveSelector(*move_list)

Thermodynamic references
------------------------
We set μ\ :sub:`Ag` from bulk Ag and μ\ :sub:`O` from ½O₂ computed with the chosen calculator.
To reduce bias from local relaxation settings during reference calculations, the calculator is
temporarily made stricter, then restored for GCMC. Finally, Δμ\ :sub:`O` is applied to explore
more oxidizing conditions.

.. code-block:: python

    from ase.build import bulk, molecule

    o2 = molecule('O2')
    ag_bulk = bulk('Ag', a=4.165)

    if hasattr(calculator, 'steps'):
        calculator.steps = 100
    if hasattr(calculator, 'fmax'):
        calculator.fmax = 0.05

    e_o2 = calculator.get_potential_energy(o2)
    e_ag = calculator.get_potential_energy(ag_bulk)

    mus = {'Ag': e_ag - 0.176, 'O': e_o2 / 2.0}
    delta_mu_O = -0.50
    mus['O'] += delta_mu_O
    T = 500.0

    if hasattr(calculator, 'steps'):
        calculator.steps = 20
    if hasattr(calculator, 'fmax'):
        calculator.fmax = 0.1

Ensemble and run
----------------
The ensemble uses metal units (eV, eV/K). The trajectory and text logs are written at
user-defined intervals; adjust these for your system size and performance goals.

.. code-block:: python

    from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble

    species = ['Ag', 'O']

    gcmc = GrandCanonicalEnsemble(
        atoms=atoms,
        cells=[cell_ag_ag, cell_ag_o],
        calculator=calculator,
        mu=mus,
        units_type='metal',
        species=species,
        temperature=T,
        move_selector=move_selector,
        outfile=f"gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_O}.out",
        trajectory_write_interval=1,
        outfile_write_interval=1,
        traj_file=f"gcmc_relax_{atoms.get_chemical_formula()}_dmu_{delta_mu_O}.xyz",
    )

    gcmc.run(1_000_000)

Interpretation
--------------
Scan several values of Δμ\ :sub:`O` to construct coverage–μ curves at fixed temperature.
Inspect acceptance ratios per move type and adjust ``min_insert``, cell height,
and move probabilities if insertions are systematically rejected or deletions dominate.
Validate that the chosen ``bottom_z`` and ``custom_height`` span the physically relevant
adsorption region without intersecting the slab interior.

Common pitfalls
---------------
- Mismatch between calculator units and ensemble settings.
- ``bottom_z`` too low or ``custom_height`` too large, causing insertions inside the slab or in vacuum.
- Overly tight ``min_insert`` leading to near-zero insertion acceptance.
- Using reference energies computed with different relaxation tolerances than the production run without justification.
