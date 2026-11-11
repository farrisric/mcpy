"""The package surface: version, exports and import side effects.
"""
import subprocess
import sys

import pytest

from mcpy.ensembles.grand_canonical_ensemble import GrandCanonicalEnsemble


def test_mcpy_exposes_version():
    import mcpy
    assert isinstance(mcpy.__version__, str) and mcpy.__version__


def test_ensembles_package_exports_grand_canonical():
    from mcpy.ensembles import GrandCanonicalEnsemble as GCE
    assert GCE is GrandCanonicalEnsemble


def test_import_does_not_mutate_matplotlib_rcparams():
    mpl = pytest.importorskip('matplotlib')
    del mpl
    code = (
        'import matplotlib\n'
        "dpi = matplotlib.rcParams['figure.dpi']\n"
        'import mcpy.moves\n'
        'import mcpy.utils\n'
        "assert matplotlib.rcParams['figure.dpi'] == dpi, "
        "matplotlib.rcParams['figure.dpi']\n"
    )
    subprocess.run([sys.executable, '-c', code], check=True)
