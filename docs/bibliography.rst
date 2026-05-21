Bibliography
============

Key references behind the algorithms implemented in `mcpy`. The list is
intentionally short; add domain-specific citations from your own work as you
adapt the library.

.. note::

   When `sphinxcontrib-bibtex` is enabled (see
   ``docs/requirements.txt``), this page will be regenerated automatically
   from ``bibliography.bib``. The entries below are placeholders so the
   page renders before that switch is made.


Hybrid Grand Canonical Monte Carlo
----------------------------------

Senftle, T. P., Janik, M. J., van Duin, A. C. T.
*A ReaxFF investigation of hydride formation in palladium nanoclusters
via Monte Carlo and molecular dynamics simulations.*
**Journal of Physical Chemistry C**, 118 (9), 4967–4981 (2014).
`doi:10.1021/jp411015a <https://doi.org/10.1021/jp411015a>`_.

The hybrid scheme — every trial insertion or deletion is followed by a
local relaxation before acceptance — is the basis of the GCMC loop in
:class:`GrandCanonicalEnsemble`.


Replica Exchange / Parallel Tempering
-------------------------------------

Swendsen, R. H., Wang, J.-S.
*Replica Monte Carlo simulation of spin-glasses.*
**Physical Review Letters**, 57 (21), 2607–2609 (1986).
`doi:10.1103/PhysRevLett.57.2607 <https://doi.org/10.1103/PhysRevLett.57.2607>`_.

The exchange acceptance rule used by :class:`ReplicaExchange` follows the
standard parallel-tempering criterion derived in this paper.


Atomic Simulation Environment
-----------------------------

Larsen, A. H., Mortensen, J. J., Blomqvist, J., et al.
*The atomic simulation environment — a Python library for working with
atoms.*
**Journal of Physics: Condensed Matter**, 29 (27), 273002 (2017).
`doi:10.1088/1361-648X/aa680e <https://doi.org/10.1088/1361-648X/aa680e>`_.

`mcpy` is built directly on ASE: every configuration is an
:class:`ase.Atoms` object, and any ASE-compatible calculator can drive the
sampling.


MACE
----

Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., Csányi, G.
*MACE: Higher order equivariant message passing neural networks for fast
and accurate force fields.*
**NeurIPS 2022**.
`arXiv:2206.07697 <https://arxiv.org/abs/2206.07697>`_.

The reference MLIP used in the bundled examples and tutorials.
