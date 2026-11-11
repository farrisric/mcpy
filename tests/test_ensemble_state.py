"""get_state and set_state of the ensembles: detached copies, swap
payloads and the restart round trip.
"""
import pickle

from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import LBFGS

from mcpy.cell import NullCell
from mcpy.ensembles import CanonicalEnsemble
from mcpy.moves import DisplacementMove, MoveSelector

from helpers import _canonical, _DirtyBailingMove, _h2, _lj_gcmc, _selector_gcmc


# Get_state returns a detached, picklable copy

def test_gcmc_get_state_is_detached_picklable_copy():
    g = _lj_gcmc(calc=EMT())
    g.atoms.calc = EMT()
    g.atoms.get_potential_energy()
    state = g.get_state()
    assert state['atoms'] is not g.atoms
    assert state['atoms'].calc is None
    pickle.dumps(state)


class _RecordingCell:
    def __init__(self):
        self.calls = 0

    def calculate_volume(self, atoms):
        self.calls += 1

    def get_volume(self):
        return 1000.0


# Set_state refreshes cell free volumes (bug: acceptance after an accepted
# replica swap used the previous configuration's volume)

def test_gcmc_set_state_recalculates_cell_volumes():
    atoms = Atoms('H2', positions=[[1, 1, 1], [3, 1, 1]], cell=[10, 10, 10], pbc=True)
    cell = _RecordingCell()
    ms = MoveSelector([1], [DisplacementMove(species=['H'], seed=1)], seed=2)
    g = _selector_gcmc(atoms, [cell], ms, mu={'H': 0.0})
    calls_before = cell.calls
    other = Atoms('H3', positions=[[1, 1, 1], [3, 1, 1], [5, 1, 1]],
                  cell=[10, 10, 10], pbc=True)
    g.set_state({'atoms': other, 'energy': -3.0, 'n_atoms': 3})
    assert cell.calls == calls_before + 1


def test_gcmc_set_state_full_dict_round_trips_counters():
    """Restart contract: get_state() -> set_state() must be lossless, or an
    appended outfile renumbers its rows from 0 after a restart."""
    g = _selector_gcmc(_h2(), [NullCell()], MoveSelector([1], [_DirtyBailingMove()]),
                       {'H': 0.0})
    partner = _selector_gcmc(_h2(), [NullCell()],
                             MoveSelector([1], [_DirtyBailingMove()]), {'H': 0.0})
    partner._step, partner.exchange_attempts, partner.exchange_successes = 40, 8, 7
    g.set_state(partner.get_state())
    assert (g._step, g.exchange_attempts, g.exchange_successes) == (40, 8, 7)


def test_canonical_set_state_full_dict_round_trips_counters(tmp_path):
    mc = _canonical(tmp_path)
    mc._step, mc.exchange_attempts, mc.exchange_successes = 12, 4, 1
    mc.set_state({'atoms': _h2(), 'energy': -1.0, 'step': 99,
                  'exchange_attempts': 40, 'exchange_successes': 39})
    assert (mc._step, mc.exchange_attempts, mc.exchange_successes) == (99, 40, 39)


def test_canonical_get_state_is_detached_copy():
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [0, 0, 2.5]], cell=[8] * 3)
    mc = CanonicalEnsemble(atoms=atoms, calculator=EMT(), optimizer=LBFGS,
                           move_selector=MoveSelector([1], [DisplacementMove(
                               species=['Cu'], seed=1)]),
                           random_seed=1, traj_file=None, outfile=None)
    mc._current_energy = mc.relax(mc.atoms)  # attaches the calculator
    state = mc.get_state()
    assert state['atoms'] is not mc.atoms
    assert state['atoms'].calc is None
