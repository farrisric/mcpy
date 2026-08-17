Spacing a replica-exchange ladder
=================================

A replica-exchange run can look perfectly healthy and still be doing nothing.
This note records how to tell, and how to space a chemical-potential ladder so
that it works. It is the ladder-design companion to
:doc:`gcmc_acceptance_convention`, and it exists because the failure modes here
are silent: no error, no warning, and output that reads as plausible physics.

.. contents::
   :local:
   :depth: 1


Read second-half acceptance, never the cumulative column
--------------------------------------------------------

Every replica starts from the same configuration, so early swaps are free: two
rungs holding identical states always exchange. Those free successes are
recorded permanently in the cumulative tally, which therefore overstates the
ladder's health for the rest of the run -- badly, and for a long time.

A five-rung CO/CuPd run reported cumulative per-slot acceptances of 28.6, 19.6,
5.9, 2.0 and 0.0 %. That reads as a ladder that mixes well at one end and
poorly at the other. In the run's *second half*, three of the four pairs
accepted **exactly zero** swaps: only pair 0-1 was still functioning, and the
other four replicas were independent single-:math:`\mu` chains.

The test needs no extra instrumentation. Attempts accumulate linearly, so
between the run's midpoint and its end the attempt count doubles. If a
cumulative percentage *halves exactly* over that span, the numerator never
changed and there were no successes at all in the second half:

.. code-block:: text

   second-half rate = 2 * cum_end - cum_mid

Slots ``0`` and ``n-1`` each belong to exactly one pair, so their columns read
those two pairs directly; interior slots average the pair on either side.

``BatchedReplicaExchange`` warns at the end of a run when the whole-run tally is
all-accept or all-reject, which catches a catastrophically mis-spaced ladder.
It cannot catch the case above, where a ladder starts healthy and dies as the
replicas differentiate. For that, run the halving test.


Why uniform spacing cannot work
-------------------------------

For a :math:`\mu` ladder at a single temperature the swap exponent is

.. math::

   \ln \frac{w'}{w} = \beta\,(\mu_i - \mu_j)(N_j - N_i)
                    = -\beta\,\Delta\mu\,\Delta N .

:math:`\Delta N` between neighbouring rungs is not a constant: it grows with
coverage, because :math:`dN/d\mu` does. A spacing chosen where the surface is
bare is therefore far too coarse once the surface fills. In the run above, at
400 K with :math:`\Delta\mu = 0.2` eV and :math:`\Delta N \approx 16`, the
exponent reaches :math:`29 \times 0.2 \times 16 \approx 93`, i.e.
:math:`p \sim 10^{-40}`. No amount of sampling recovers that pair.


The spacing rule
----------------

Grand-canonical fluctuation-dissipation gives the width of each rung's
:math:`N` distribution,

.. math::

   \sigma_N^2 = \frac{1}{\beta}\frac{dN}{d\mu} ,

and neighbouring rungs exchange when their distributions overlap, i.e. when
:math:`\beta\,\Delta\mu\,\sigma_N \sim 1`. Substituting:

.. math::

   \Delta\mu(\mu) = \frac{1}{\sqrt{\beta\,dN/d\mu}} ,
   \qquad
   n_\text{gaps} = \int \sqrt{\beta\,\frac{dN}{d\mu}}\; d\mu ,
   \qquad
   n_\text{rungs} = n_\text{gaps} + 1 .

Use :math:`dN/d\mu` from a measured isotherm, **not** the per-rung
:math:`\sigma_N` observed in an unconverged run: a stuck rung reports a
:math:`\sigma_N` far below equilibrium and a drifting one far above, because the
"fluctuation" is drift.

Histogram reweighting is not an alternative route to this. Reweighting requires
overlap between adjacent rungs, and a mis-spaced ladder has none -- that absence
is the disease being diagnosed.

.. code-block:: python

   import numpy as np

   def acceptance_equalized_mu_ladder(mu, n_ads, temperature, mu_min, mu_max):
       """Rung positions for a chemical-potential ladder of even acceptance.

       Places rungs where the cumulative integral of sqrt(beta dN/dmu) crosses
       successive integers, so every neighbouring pair has comparable
       distribution overlap and therefore comparable swap acceptance.

       Parameters
       ----------
       mu, n_ads : array_like
           A measured isotherm: chemical potentials and mean adsorbate counts.
       temperature : float
           Ladder temperature in K (all rungs share it).
       mu_min, mu_max : float
           Range to span.

       Returns
       -------
       ndarray
           Rung chemical potentials, ascending. Spacing widens where the
           isotherm is flat and tightens where it is steep.
       """
       beta = 1.0 / (8.617333e-5 * temperature)
       mu, n_ads = np.asarray(mu, float), np.asarray(n_ads, float)
       order = np.argsort(mu)             # desorption scans arrive descending
       mu, n_ads = mu[order], n_ads[order]
       slope = np.diff(n_ads) / np.diff(mu)          # dN/dmu per interval
       midpoints = 0.5 * (mu[1:] + mu[:-1])

       fine = np.linspace(mu_min, mu_max, 2001)
       # Clip noise-induced negative dN/dmu to zero: a NaN here would
       # propagate into ``cumulative`` and crash np.interp cryptically.
       density = np.sqrt(beta * np.maximum(np.interp(fine, midpoints, slope),
                                           0.0))
       cumulative = np.concatenate([[0.0], np.cumsum(
           0.5 * (density[1:] + density[:-1]) * np.diff(fine))])
       # linspace, not arange: the ladder must reach mu_max, so the last gap
       # shrinks rather than the top rung being dropped.
       targets = np.linspace(0.0, cumulative[-1],
                             int(np.ceil(cumulative[-1])) + 1)
       return np.interp(targets, cumulative, fine)


Calibration
-----------

The rule above is derived from the exponent at the *mean* :math:`\Delta N`, but
the acceptance that matters is :math:`\langle \min(1, \cdot) \rangle` over both
rungs' fluctuations. A favourable fluctuation accepts outright, so the realised
acceptance is higher than the point estimate suggests -- for the CO/CuPd system
the rule aimed at ~25 % and delivered ~40 %.

Treat the rule as a safe upper bound on rung count, then trim once a run has
measured the real acceptance. A ladder derived from an unconverged isotherm is
also a lower bound on the rungs eventually needed, since :math:`dN/d\mu`
steepens as coverage converges. Those two biases push in opposite directions;
re-derive from each run rather than trusting one calculation.


Worked example: CO on a CuPd nanoparticle
-----------------------------------------

Cu\ :sub:`375`\ Pd\ :sub:`30` at 400 K, :math:`\Delta\mu` spanning -1.8 to -1.0
eV, ``AlchemiFCalculator``.

.. list-table::
   :header-rows: 1

   * - Interval (eV)
     - :math:`dN/d\mu`
     - Spacing
     - Rungs
   * - -1.8 .. -1.6
     - 8.5
     - 0.064 eV
     - 3
   * - -1.6 .. -1.4
     - 45.1
     - 0.028 eV
     - 7
   * - -1.4 .. -1.2
     - 54.5
     - 0.025 eV
     - 8
   * - -1.2 .. -1.0
     - 88.9
     - 0.020 eV
     - 10

29 rungs, against the 5 that a uniform 0.2 eV ladder would use. Measured
second-half acceptance: minimum 18 %, median 40 %, maximum 65 %, with no dead
pair -- and a smooth, strictly monotonic isotherm from 0.1 to 57 CO.

Two practical consequences of going wide:

- **Chunking becomes mandatory.** 29 replicas of ~460 atoms is ~13k atoms,
  several times the whole-batch relaxation ceiling. Pass ``chunk_size`` so peak
  memory follows the largest chunk rather than the replica count.
- **It is faster, not slower, in absolute terms.** A narrow ladder leaves the
  GPU idle. Five replicas ran at ~43 % utilisation; 29 replicas at
  ``chunk_size=8`` reached 94-98 % and 15.6 GB of 32.6 GB.

Once the ladder is healthy, convergence becomes the binding constraint rather
than the sampler. Chain runs with ``--init-dir`` pointing at the previous
output and watch the tail-half drift per rung, not just the acceptance.
