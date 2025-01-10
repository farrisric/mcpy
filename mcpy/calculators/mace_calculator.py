import torch
from mace.tools import torch_tools, utils
from mace import data


class Calculator():
    def __init__(self) -> None:
        torch_tools.set_default_dtype('float64')
        self.model = torch.load('/work/g15farris/mace-large-density-agnesi-stress.model',
                                map_location='cpu')
        self.model = self.model.to('cpu')
        for param in self.model.parameters():
            param.requires_grad = False
        self.z_table = utils.AtomicNumberTable([int(z) for z in self.model.atomic_numbers])

    def get_atom_data(self, atoms):
        atom_data = data.AtomicData.from_config(
                        data.config_from_atoms(atoms), z_table=self.z_table,
                        cutoff=float(self.model.r_max), heads=None
        )
        atom_data = atom_data.to_dict()
        atom_data['batch'] = torch.zeros(len(atoms), dtype=torch.long)
        atom_data['head'] = torch.tensor([0], dtype=torch.long)
        atom_data['ptr'] = torch.tensor([0, atom_data['batch'].shape[0]])
        return atom_data

    def get_potential_energy(self, atoms):
        atom_data = self.get_atom_data(atoms)
        with torch.no_grad():
            energy = self.model(atom_data, compute_stress=False, compute_force=False)['energy']
        return energy.item()
