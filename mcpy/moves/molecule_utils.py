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
    return _coms(atoms, [np.asarray(members)])[0]


def _coms(atoms, groups):
    """Centers of mass of several index groups in one vectorized pass
    (one ``find_mic`` and one ``wrap_positions`` for all groups)."""
    if not groups:
        return np.empty((0, 3))
    idx = np.concatenate(groups)
    sizes = np.fromiter((len(g) for g in groups), dtype=int, count=len(groups))
    starts = np.concatenate(([0], np.cumsum(sizes)[:-1]))
    pos = atoms.positions[idx]
    ref = np.repeat(pos[starts], sizes, axis=0)
    diff = pos - ref
    use_mic = bool(np.any(np.asarray(atoms.pbc)))
    if use_mic:
        diff = find_mic(diff, atoms.cell, pbc=atoms.pbc)[0]
    masses = atoms.get_masses()[idx]
    total = np.add.reduceat(masses, starts)
    com = pos[starts] + np.add.reduceat(diff * masses[:, None], starts) / total[:, None]
    if use_mic:
        com = wrap_positions(com, atoms.cell, pbc=atoms.pbc)
    return com


def molecule_groups(atoms):
    """``{id: member indices}`` for every molecule id >= 0, in id order."""
    ids = atoms.arrays.get('molecule_id')
    if ids is None:
        return {}
    order = np.argsort(ids, kind='stable')
    uniq, starts = np.unique(ids[order], return_index=True)
    return {int(mid): members for mid, members
            in zip(uniq, np.split(order, starts[1:])) if mid >= 0}


def find_molecules(atoms, template_symbols, cell=None, proposal_region=False):
    """Index groups of whole molecules matching ``template_symbols``.

    ``template_symbols`` is the sorted list of the template's chemical
    symbols. When ``cell`` is given, only molecules whose center of mass
    the cell counts as exchangeable (``cell.is_point_exchangeable``) are
    returned; ``None`` skips the spatial filter (used by the grand-potential
    bookkeeping). The exchangeable region is not always the proposal region:
    see :meth:`mcpy.cell.BaseCell.is_point_exchangeable`.
    ``proposal_region=True`` tests ``cell.is_point_inside`` instead, the
    region displacement moves keep molecules in.
    """
    symbols = np.asarray(atoms.get_chemical_symbols())
    template = list(template_symbols)
    groups = [m for m in molecule_groups(atoms).values()
              if sorted(symbols[m]) == template]
    if cell is None or not groups:
        return groups
    predicate = cell.is_point_inside if proposal_region else cell.is_point_exchangeable
    return [m for m, com in zip(groups, _coms(atoms, groups)) if predicate(com)]


def quaternion_matrix(w, x, y, z):
    """Rotation matrix of the unit quaternion ``(w, x, y, z)``."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def random_rotation_matrix(rng):
    """Uniform random 3x3 rotation matrix (normalized 4-Gaussian quaternion)."""
    q = np.array([rng.get_gaussian() for _ in range(4)])
    q /= np.linalg.norm(q)
    return quaternion_matrix(*q)
