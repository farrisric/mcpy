"""SetUnits (LJ and metal units) and the seeding of the random streams.
"""
import numpy as np
import pytest

from mcpy.moves import DisplacementMove
from mcpy.utils import RandomNumberGenerator
from mcpy.utils.set_unit_constant import SetUnits


# LJ unit system defines beta (bug: AttributeError on first trial move)

def test_lj_units_define_beta():
    assert SetUnits('LJ', temperature=1.0, species=['X']).beta == 1.0
    assert SetUnits('LJ', temperature=2.0, species=['X']).beta == pytest.approx(0.5)


# Numpy integer seeds

def test_numpy_integer_seed():
    a = RandomNumberGenerator(np.int64(5)).get_uniform()
    assert a == RandomNumberGenerator(5).get_uniform()
    DisplacementMove(['H'], seed=np.int32(3))
