from ase import Atoms
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.cell import Cell


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
