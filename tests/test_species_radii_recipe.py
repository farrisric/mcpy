"""The r_relax recipe of docs/species_radii.rst takes the most frequent
relaxed distance (the tallest histogram bin), so a small spurious bump at a
short distance does not set the radius. A percentile rule would."""
import textwrap
from pathlib import Path

import numpy as np

DOC = Path(__file__).resolve().parent.parent / 'docs' / 'species_radii.rst'


def _recipe():
    """The lines of the docs code block after the relaxation loop."""
    block = DOC.read_text().split('.. code-block:: python\n\n', 1)[1]
    block = textwrap.dedent(block.split('\n.. figure::', 1)[0])
    return block.split('distances.append(d.min())\n', 1)[1]


def test_r_relax_is_the_tallest_bin_not_a_spurious_bump():
    rng = np.random.default_rng(0)
    distances = np.concatenate([
        rng.normal(1.0, 0.1, 6),     # spurious short-distance bump, 6 %
        rng.normal(2.11, 0.02, 60),  # the binding peak
        rng.uniform(3.0, 5.5, 34),   # trials that did not bind
    ]).tolist()
    namespace = {'np': np, 'distances': distances}
    exec(_recipe(), namespace)
    assert abs(namespace['r_relax'] - 2.11) < 0.05
