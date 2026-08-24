from ase import Atoms

import numpy as np


def get_p_at_support(support: Atoms, particle: Atoms,
                     contact_surface: str = '100', gap: float = 2.0,
                     vacuum_z: float = 10.0) -> Atoms:
    """
    Place a nanoparticle on top of a support. The passed ``particle`` is
    modified in place (cell cleared, rotated/translated into position); the
    support is copied.

    Parameters
    ----------
    support : Atoms
        The slab/support (its cell a,b are reused; z-PBC disabled).
    particle : Atoms
        The nanoparticle to place. Mutated in place.
    contact_surface : {'100','111'}
        Which facet should contact the support.
    gap : float
        Vertical clearance between support top z and particle bottom z (Å).
    vacuum_z : float
        Vacuum padding added along z after combining (Å).

    Returns
    -------
    Atoms
        Combined system with tags: 0 = support, 1 = particle.
    """
    com_xy = support.get_center_of_mass()[:2]
    surface_z = float(np.max(support.positions[:, 2]))

    particle.cell = None
    particle.translate(-particle.get_center_of_mass())

    # Optional orientation: make a (111) facet face down
    if contact_surface == '111':
        # Simple deterministic orientation; tweak if you need a specific in-plane rotation
        particle.rotate('x', 45, rotate_cell=False)
        particle.rotate('y', 35, rotate_cell=False)
        particle.rotate('z', 90, rotate_cell=False)

    # Move particle over the support in XY
    particle.translate([com_xy[0], com_xy[1], 0.0])

    # Lift particle so its lowest atom sits just above the support top, leaving a gap
    min_z = float(np.min(particle.positions[:, 2]))
    dz = (surface_z + gap) - min_z
    particle.translate([0.0, 0.0, dz])

    sup = support.copy()
    sup.set_tags(0)
    particle.set_tags(1)

    atoms = sup + particle
    atoms.set_cell(support.cell, scale_atoms=False)
    atoms.set_pbc((True, True, False))
    atoms.center(vacuum=vacuum_z, axis=2)  # add 10 Å vacuum along z

    return atoms
