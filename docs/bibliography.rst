Bibliography
============

Key references behind the methods that `mcpy` implements.


Hybrid Grand Canonical Monte Carlo
----------------------------------

Senftle, T. P., Janik, M. J., van Duin, A. C. T.
*A ReaxFF investigation of hydride formation in palladium nanoclusters
via Monte Carlo and molecular dynamics simulations.*
**The Journal of Physical Chemistry C**, 118 (9), 4967-4981 (2014).
`doi:10.1021/jp411015a <https://doi.org/10.1021/jp411015a>`_.

The hybrid scheme, in which every trial insertion or deletion is relaxed
before the acceptance test, is the basis of the GCMC loop in
:class:`~mcpy.ensembles.GrandCanonicalEnsemble`.


GCMC for surface phase diagrams
-------------------------------

Wexler, R. B., Qiu, T., Rappe, A. M.
*Automatic prediction of surface phase diagrams using ab initio grand
canonical Monte Carlo.*
**The Journal of Physical Chemistry C**, 123 (4), 2321-2328 (2019).
`doi:10.1021/acs.jpcc.8b11093 <https://doi.org/10.1021/acs.jpcc.8b11093>`_.

Grand-canonical sampling restricted to a region near the surface, the
approach of the slab examples (:doc:`examples/gcmc_custom_cell`).


Replica exchange and parallel tempering
---------------------------------------

Geyer, C. J.
*Markov chain Monte Carlo maximum likelihood.*
In: Computing Science and Statistics, Proceedings of the 23rd Symposium on
the Interface, pp. 156-163. Interface Foundation of North America (1991).

Hukushima, K., Nemoto, K.
*Exchange Monte Carlo method and application to spin glass simulations.*
**Journal of the Physical Society of Japan**, 65 (6), 1604-1608 (1996).
`doi:10.1143/JPSJ.65.1604 <https://doi.org/10.1143/JPSJ.65.1604>`_.

The swap acceptance rule of :class:`~mcpy.ensembles.ReplicaExchange` and
:class:`~mcpy.ensembles.BatchedReplicaExchange` is the parallel-tempering
(exchange Monte Carlo) criterion of these papers, extended to a
chemical-potential ladder.

Swendsen, R. H., Wang, J.-S.
*Replica Monte Carlo simulation of spin-glasses.*
**Physical Review Letters**, 57 (21), 2607-2609 (1986).
`doi:10.1103/PhysRevLett.57.2607 <https://doi.org/10.1103/PhysRevLett.57.2607>`_.

The original replica Monte Carlo method, a precursor of replica exchange.


Atomic Simulation Environment
-----------------------------

Larsen, A. H., Mortensen, J. J., Blomqvist, J., et al.
*The atomic simulation environment: a Python library for working with
atoms.*
**Journal of Physics: Condensed Matter**, 29 (27), 273002 (2017).
`doi:10.1088/1361-648X/aa680e <https://doi.org/10.1088/1361-648X/aa680e>`_.

`mcpy` is built directly on ASE: every configuration is an ``ase.Atoms``
object, and any ASE-compatible calculator can drive the sampling.


MACE
----

Batatia, I., Kovács, D. P., Simm, G. N. C., Ortner, C., Csányi, G.
*MACE: Higher order equivariant message passing neural networks for fast
and accurate force fields.*
**NeurIPS 2022**.
`arXiv:2206.07697 <https://arxiv.org/abs/2206.07697>`_.

The machine-learned potential of the GPU examples and the tutorial.


Thesis
------

Farris, R.
*Development and Application of Advanced Computational Methods for
Predicting the Structure and Properties of Nanoparticles.*
PhD thesis, Universitat de Barcelona (2025).
`hdl:10803/697698 <https://hdl.handle.net/10803/697698>`_.

The doctoral work in which `mcpy` was developed; the exclusion-radius
calibration (:doc:`species_radii`) follows it.
