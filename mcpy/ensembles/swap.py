"""The replica-exchange swap rule, shared by ``ReplicaExchange`` (MPI) and
``BatchedReplicaExchange``.

A replica describes itself to the rule through its ``swap_view()``: a dict
with ``beta``, ``energy``, and for a grand-canonical replica also ``mu``,
``lambda_dbs`` and ``counts``. ``counts`` holds the particle count of the
replica's own configuration for each mu species, counted the way its own
acceptance counts them (molecules for molecular species, free atoms
otherwise). Both drivers call the same function, so they cannot drift apart,
and the MPI driver counts molecules exactly as the batched one does.
"""
import numpy as np


def swap_log_ratio(x, y) -> float:
    """ln(w'/w) for swapping configuration X (slot i, view ``x``) with
    configuration Y (slot j, view ``y``).

    Slot k weights a configuration Z by
    exp(-beta_k Phi_Z^(k)) prod_s Lambda_s(T_k)^(-3 N_Z,s), with
    Phi_Z^(k) = E_Z - sum_s mu_s^(k) N_Z,s scored with slot k's own mu. So

        ln(w'/w) = beta_i (Phi_X^(i) - Phi_Y^(i)) + beta_j (Phi_Y^(j) - Phi_X^(j))
                   + 3 sum_s (N_X,s - N_Y,s) ln(Lambda_i,s / Lambda_j,s)
                 = (beta_i - beta_j) dE - sum_s (beta_i mu_i,s - beta_j mu_j,s) dN_s
                   + 3 sum_s dN_s ln(Lambda_i,s / Lambda_j,s),

    with dE = E_X - E_Y and dN_s = N_X,s - N_Y,s. The code uses the second
    form: only differences enter, so substrate atoms cancel, and on a mu
    ladder (beta_i == beta_j) the energies drop out. The cross terms keep the
    rule right on a T ladder whose rungs set their own mu(T). On a shared mu
    it is (beta_j - beta_i)(Phi_Y - Phi_X); on a shared T it is
    beta sum_s (mu_i,s - mu_j,s)(N_Y,s - N_X,s), which is not zero (the
    short T-ladder form would accept every swap on a mu ladder).

    Lambda depends on T only, so the de Broglie term is zero on a mu ladder
    and in LJ units (Lambda = 1). A replica without mu (``CanonicalEnsemble``)
    exchanges nothing with a reservoir and scores bare E.
    """
    mu_i, mu_j = x.get('mu') or {}, y.get('mu') or {}
    if set(mu_i) != set(mu_j):
        raise ValueError(
            f'replicas exchange different species ({sorted(mu_i)} and '
            f'{sorted(mu_j)}); every rung needs a mu for the same species')
    beta_i, beta_j = x['beta'], y['beta']
    # inf - inf and 0 * inf give NaN, which the caller rejects.
    delta = (beta_i - beta_j) * (x['energy'] - y['energy'])
    for specie in mu_i:
        dn = x['counts'][specie] - y['counts'][specie]
        if not dn:
            continue
        delta -= (beta_i * mu_i[specie] - beta_j * mu_j[specie]) * dn
        if beta_i != beta_j:
            delta += 3.0 * dn * np.log(x['lambda_dbs'][specie]
                                       / y['lambda_dbs'][specie])
    return float(delta)


def swap_probability(x, y, logger, label='') -> float:
    """min(1, exp(``swap_log_ratio(x, y)``)), and 0 with a warning when the
    exponent is not finite. ``label`` names the pair in the log."""
    delta = swap_log_ratio(x, y)
    if not np.isfinite(delta):
        # min(1, exp(nan)) is 1: a diverged energy would always swap, and a
        # silent rejection would hide a dead rung.
        logger.warning(
            'swap %s: non-finite acceptance exponent (delta=%r); rejecting. '
            'A NaN/inf replica energy usually means a diverged relaxation.',
            label, delta)
        return 0.0
    # exp() overflows to inf for a large positive delta; short-circuit.
    p = 1.0 if delta >= 0.0 else float(np.exp(delta))
    logger.debug('swap %s: beta=(%.4e, %.4e) E=(%.6f, %.6f) counts=(%s, %s) '
                 'delta=%.4f p=%.4f', label, x['beta'], y['beta'], x['energy'],
                 y['energy'], x.get('counts'), y.get('counts'), delta, p)
    return p
