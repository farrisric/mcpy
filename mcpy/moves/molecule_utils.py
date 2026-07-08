"""Bookkeeping helpers for rigid-molecule GCMC moves.

Molecule membership is stored in the per-atom int array
``atoms.arrays['molecule_id']``: ``-1`` means not part of any molecule,
otherwise all member atoms share one ID. Because the array lives in
``atoms.arrays`` it is snapshot/rolled back by the ensembles for free,
written to extxyz trajectories, and survives restarts.

A molecule's species is identified by composition: sorted member symbols
equal the sorted template symbols. Two molecular species with identical
composition (isomers) therefore cannot coexist; ``SetUnits`` rejects such
a configuration at construction.
"""
import numpy as np
from ase.geometry import find_mic, wrap_positions


def molecule_com(atoms, members):
    """Mass-weighted center of mass of ``atoms[members]``.

    Minimum-image aware: member positions are unwrapped relative to the
    first member before averaging, because ``atoms.wrap()`` on accepted
    moves can split a molecule across periodic boundaries. The result is
    wrapped back into the box so region-cell inside tests see a point in
    the primary cell.
    """
    pos = atoms.positions[members]
    ref = pos[0]
    diff = pos - ref
    use_mic = bool(np.any(np.asarray(atoms.pbc)))
    if use_mic:
        diff = find_mic(diff, atoms.cell, pbc=atoms.pbc)[0]
    masses = atoms.get_masses()[members]
    com = ref + (diff * masses[:, None]).sum(axis=0) / masses.sum()
    if use_mic:
        com = wrap_positions([com], atoms.cell, pbc=atoms.pbc)[0]
    return com


def find_molecules(atoms, template_symbols, cell=None):
    """Index groups of whole molecules matching ``template_symbols``.

    ``template_symbols`` is the sorted list of the template's chemical
    symbols. When ``cell`` is given, only molecules whose center of mass
    lies inside it (``cell.is_point_inside``) are returned; ``None`` skips
    the spatial filter (used by the grand-potential bookkeeping).
    """
    ids = atoms.arrays.get('molecule_id')
    if ids is None:
        return []
    symbols = np.asarray(atoms.get_chemical_symbols())
    groups = []
    for mid in np.unique(ids):
        if mid < 0:
            continue
        members = np.where(ids == mid)[0]
        if sorted(symbols[members]) != list(template_symbols):
            continue
        if cell is not None and not cell.is_point_inside(molecule_com(atoms, members)):
            continue
        groups.append(members)
    return groups


def random_rotation_matrix(rng):
    """Uniform random 3x3 rotation matrix (normalized 4-Gaussian quaternion)."""
    q = np.array([rng.get_gaussian() for _ in range(4)])
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
