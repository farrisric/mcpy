"""Debug output for MD-based trial moves: one extended-XYZ frame per MD step."""
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write


def write_md_frame(path: str, atoms: Atoms, positions, velocities, energy: float,
                   forces, *, md_run: int, md_step: int, dt: float,
                   temperature: float) -> None:
    """Append one MD frame to ``path``.

    ``atoms`` supplies symbols, cell, constraints and per-atom arrays.
    ``velocities`` are in ASE units (A per ASE time unit); they are stored as
    ``momenta``, so ``read(path).get_velocities()`` gives them back. ``dt`` is
    the timestep in fs and ``temperature`` the target temperature in K. The
    info line carries the instantaneous ``T_K`` and ``epot``/``ekin``/``etot``
    in eV.
    """
    frame = atoms.copy()
    frame.positions = positions
    frame.set_velocities(velocities)
    frame.calc = SinglePointCalculator(frame, energy=energy, forces=forces)
    ekin = frame.get_kinetic_energy()
    frame.info.update(
        md_run=md_run, md_step=md_step, time_fs=md_step * dt,
        T_target_K=temperature, T_K=frame.get_temperature(),
        epot=energy, ekin=ekin, etot=energy + ekin)
    write(path, frame, format='extxyz', append=True)
