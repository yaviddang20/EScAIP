"""
Convert a dictionary from the mptrj dataset to a PyG Data object.
Ref: https://github.com/facebookresearch/JMP/blob/main/src/jmp/datasets/finetune/mat_bench.py
"""

import torch
import numpy as np
from torch_geometric.data import Data
from pymatgen.core.structure import Structure


def mptrj_dict_to_pyg_data(dic, sid):
    structure = Structure.from_dict(dic["structure"])

    # Input
    atomic_numbers = torch.tensor(
        [site.specie.number for site in structure], dtype=torch.long
    )  # natoms
    cell = torch.tensor(structure.lattice.matrix, dtype=torch.float).unsqueeze(
        dim=0
    )  # 1 3 3
    pos = torch.tensor(
        np.array([site.coords for site in structure]), dtype=torch.float
    )  # natoms 3

    # Output
    energy = torch.tensor(dic["uncorrected_total_energy"], dtype=torch.float)
    force = torch.tensor(dic["force"], dtype=torch.float)  # natoms 3
    corrected_total_energy = torch.tensor(
        dic["corrected_total_energy"], dtype=torch.float
    )
    energy_per_atom = torch.tensor(dic["energy_per_atom"], dtype=torch.float)
    ef_per_atom = torch.tensor(dic["ef_per_atom"], dtype=torch.float)
    e_per_atom_relaxed = torch.tensor(dic["e_per_atom_relaxed"], dtype=torch.float)
    ef_per_atom_relaxed = torch.tensor(dic["ef_per_atom_relaxed"], dtype=torch.float)
    stress = torch.tensor(dic["stress"], dtype=torch.float)
    magmom = torch.tensor(dic["magmom"], dtype=torch.float)
    bandgap = torch.tensor(dic["bandgap"], dtype=torch.float)

    mp_id = dic["mp_id"]

    data = Data(
        atomic_numbers=atomic_numbers,
        pos=pos,
        cell=cell,
        y=energy,
        force=force,
        corrected_total_energy=corrected_total_energy,
        energy_per_atom=energy_per_atom,
        ef_per_atom=ef_per_atom,
        e_per_atom_relaxed=e_per_atom_relaxed,
        ef_per_atom_relaxed=ef_per_atom_relaxed,
        stress=stress,
        magmom=magmom,
        bandgap=bandgap,
        mp_id=mp_id,
        sid=sid,
    )
    return data
