from ase import Atoms
from ase.geometry import get_layers

def find_surface_indices(atoms: Atoms, tolerance: float = 0.1) -> list:
    """
    Find the surface indices of an atoms object.

    Parameters:
    atoms (Atoms): The atoms object.
    tolerance (float): The tolerance for identifying surface atoms.

    Returns:
    list: A list of surface atom indices.
    """
    # Get the layers of atoms
    layers, _ = get_layers(atoms, tolerance=tolerance)
    
    # Find the maximum layer index which corresponds to the surface
    max_layer = max(layers)
    
    # Get the indices of atoms in the surface layer
    surface_indices = [index for index, layer in enumerate(layers) if layer == max_layer]
    
    return surface_indices

    def find_nanoparticle_surface_indices(atoms: Atoms, tolerance: float = 0.1) -> list:
        """
        Find the surface indices of a nanoparticle atoms object.

        Parameters:
        atoms (Atoms): The atoms object representing the nanoparticle.
        tolerance (float): The tolerance for identifying surface atoms.

        Returns:
        list: A list of surface atom indices.
        """
        # Get the layers of atoms
        layers, _ = get_layers(atoms, tolerance=tolerance)
        
        # Find the maximum layer index which corresponds to the surface
        max_layer = max(layers)
        
        # Get the indices of atoms in the surface layer
        surface_indices = [index for index, layer in enumerate(layers) if layer == max_layer]
        
        return surface_indices