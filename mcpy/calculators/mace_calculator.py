import torch
from mace.tools import torch_tools, utils
from mace import data
from ase import Atoms


class MACECalculator:
    def __init__(self, model_paths: str, device: str = 'cpu') -> None:
        """
        Initialize the Calculator with the given model path and device.

        :param model_paths: Path to the model file.
        :param device: Device to load the model on ('cpu' or 'cuda').
        """
        torch_tools.set_default_dtype('float64')
        self.model = torch.load(model_paths, map_location=device).to(device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])

    def get_atom_data(self, atoms: Atoms) -> dict:
        """
        Convert ASE Atoms object to the required atomic data format.

        :param atoms: ASE Atoms object.
        :return: Dictionary containing atomic data.
        """
        atom_data = data.AtomicData.from_config(
            data.config_from_atoms(atoms), z_table=self.z_table,
            cutoff=float(self.model.r_max), heads=None
        ).to_dict()
        atom_data.update({
            'batch': torch.zeros(len(atoms), dtype=torch.long),
            'head': torch.tensor([0], dtype=torch.long),
            'ptr': torch.tensor([0, len(atoms)])
        })
        return atom_data

    def get_potential_energy(self, atoms: Atoms) -> float:
        """
        Calculate the potential energy of the given atoms.

        :param atoms: ASE Atoms object.
        :return: Potential energy as a float.
        """
        atom_data = self.get_atom_data(atoms)
        with torch.no_grad():
            energy = self.model(atom_data, compute_stress=False, compute_force=False)['energy']
        return energy.item()
