from .base_calculator import BaseCalculator

# The backend calculators import lazily (PEP 562): mace-torch and nvalchemi
# pull in torch, which costs seconds and, in a misconfigured env, kills the
# process with SIGBUS. Accessing one without its backend raises the real
# ImportError (install with: pip install -e .[mace] or .[alchemi]).
_LAZY = {
    'MACE_F_Calculator': 'mace_f_calculator',
    'AlchemiCalculator': 'alchemi',
    'AlchemiFCalculator': 'alchemi',
}

__all__ = ['BaseCalculator', *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        return getattr(import_module(f'.{_LAZY[name]}', __name__), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted({*globals(), *__all__})
