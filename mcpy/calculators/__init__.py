from .base_calculator import BaseCalculator

__all__ = ['BaseCalculator']

try:
    from .mace_calculator import MACECalculator
    from .mace_f_calculator import MACE_F_Calculator
    __all__ += ['MACECalculator', 'MACE_F_Calculator']
except ImportError:
    # mace-torch is an optional backend; install with: pip install -e .[mace]
    pass

try:
    from .alchemi_calculator import AlchemiCalculator
    from .alchemi_f_calculator import AlchemiFCalculator
    __all__ += ['AlchemiCalculator', 'AlchemiFCalculator']
except ImportError:
    # nvalchemi is an optional backend; skip if not installed.
    pass
