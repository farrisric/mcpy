from .random_number_generator import RandomNumberGenerator

__all__ = [
    'RandomNumberGenerator',
    'analyze_phase_diagram_results',
    'plot_phase_diagram',
]


def __getattr__(name):
    # Lazy re-export (PEP 562): phase_diagram pulls in matplotlib, which the
    # MC core (moves import mcpy.utils) must not require or configure.
    if name in ('analyze_phase_diagram_results', 'plot_phase_diagram'):
        from . import phase_diagram
        return getattr(phase_diagram, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
