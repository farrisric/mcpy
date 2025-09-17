from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.cell import Cell

import numpy as np
import math


def get_volume(box: list) -> float:
    """
    Get the volume of a box.
    """
    box = Cell(box)
    return box.volume


def find_surface_indices(atoms: Atoms, tolerance: float = 0.1) -> list:
    """
    Find the surface indices of an atoms object.

    Parameters:
    atoms (Atoms): The atoms object.
    tolerance (float): The tolerance for identifying surface atoms.

    Returns:
    list: A list of surface atom indices.
    """
    # Create a neighbor list
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = NeighborList(cutoffs, self_interaction=False, bothways=True)
    neighbor_list.update(atoms)

    surface_indices = []
    for i in range(len(atoms)):
        indices, offsets = neighbor_list.get_neighbors(i)
        if len(indices) < 12:
            print(len(indices))
            surface_indices.append(i)

    return surface_indices


def sphere_volume(radius):
    return (4/3) * math.pi * radius**3


def total_volume(spheres):
    total_vol = sum(sphere_volume(radius) for radius in spheres)
    return total_vol


def overlap_volume(r1, r2, d):
    """Calculate the volume of overlap between two spheres of radius r1, r2 and distance d."""
    if d >= r1 + r2:
        return 0  # No overlap
    if d <= abs(r1 - r2):
        return (4/3) * math.pi * min(r1, r2)**3

    part1 = (math.pi * (r1 + r2 - d)**2 * (d**2 + 2*d*(r1 + r2) - 3*(r1 - r2)**2)) / (12 * d)
    part2 = (math.pi * (r1 + r2 - d)**2 * (d**2 + 2*d*(r1 + r2) - 3*(r1 - r2)**2)) / (12 * d)
    return part1 + part2


def total_volume_with_overlap(spheres, positions):
    total_vol = 0
    num_spheres = len(spheres)
    for i in range(num_spheres):
        total_vol += sphere_volume(spheres[i])
    print('Total volue of Spheres: ', total_vol)

    for i in range(num_spheres):
        for j in range(i + 1, num_spheres):
            d = math.dist(positions[i], positions[j])  # Distance between sphere centers
            total_vol -= overlap_volume(spheres[i], spheres[j], d)
    return total_vol

def get_p_at_support(support: Atoms, particle: Atoms,
                     contact_surface: str = '100', gap: float = 2.0) -> Atoms:
    """
    Place an Ag truncated octahedron on top of a support.
    
    Parameters
    ----------
    support : Atoms
        The slab/support (its cell a,b are reused; z-PBC disabled).
    layers : int
        Octahedron size parameter for ASE Octahedron.
    cutoff : float
        Cutoff for ASE Octahedron (Å).
    contact_surface : {'100','111'}
        Which facet should contact the support.
    gap : float
        Vertical clearance between support top z and particle bottom z (Å).
        
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
    atoms.center(vacuum=10.0, axis=2)  # add 10 Å vacuum along z

    return atoms
