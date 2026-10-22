"""Fit the Beta (2-component Dirichlet) prior delta of the native proportion.

fit_r reproduces celda, which calls MCMCprecision::fit_dirichlet with its
defaults: the Minka (2000) fixed point
alpha <- inv_digamma(digamma(sum(alpha)) + mean(log x)), with zeros shifted by
1% of the smallest positive value, the method-of-moments estimate (at least
0.01) as the start, and a stop when no component changes by more than
abstol = 0.1.

It takes the native proportions p of the cells (the contamination is 1 - p)
and returns the previous delta when the data cannot give an estimate.
"""

import numpy as np
from numba import njit

from ._special import digamma, inv_digamma

# MCMCprecision::fit_dirichlet defaults.
_R_ABSTOL = 0.1
_R_MAXIT = 100000


@njit(cache=True)
def fit_r(p, delta):
    """Celda's delta: MCMCprecision::fit_dirichlet behaviour with its defaults."""
    n = len(p)
    # Shift zeros by 1% of the smallest positive entry of (p, 1 - p).
    smallest = np.inf
    has_zero = False
    for i in range(n):
        for x in (p[i], 1.0 - p[i]):
            if x <= 0.0:
                has_zero = True
            elif x < smallest:
                smallest = x
    const = 0.01 * smallest if has_zero else 0.0

    log0 = 0.0
    log1 = 0.0
    m0 = 0.0
    m1 = 0.0
    sq0 = 0.0
    sq1 = 0.0
    for i in range(n):
        x0 = (p[i] + const) / (1.0 + 2.0 * const)
        x1 = (1.0 - p[i] + const) / (1.0 + 2.0 * const)
        total = x0 + x1
        x0 /= total
        x1 /= total
        log0 += np.log(x0)
        log1 += np.log(x1)
        m0 += x0
        m1 += x1
        sq0 += x0 * x0
        sq1 += x1 * x1
    log0 /= n
    log1 /= n
    m0 /= n
    m1 /= n
    sq0 /= n
    sq1 /= n

    # Proportions without spread (up to rounding) have no finite estimate.
    if min(sq0 - m0 * m0, sq1 - m1 * m1) <= 1e-14 * m0 * m1:
        return delta
    a0 = max(0.01, (m0 - sq0) / (sq0 - m0 * m0) * m0)
    a1 = max(0.01, (m1 - sq1) / (sq1 - m1 * m1) * m1)
    if not (np.isfinite(a0) and np.isfinite(a1)):
        return delta
    for _ in range(_R_MAXIT):
        s = digamma(a0 + a1)
        b0 = inv_digamma(s + log0, 5)
        b1 = inv_digamma(s + log1, 5)
        if not (np.isfinite(b0) and np.isfinite(b1)):
            return delta
        step = max(abs(b0 - a0), abs(b1 - a1))
        a0 = b0
        a1 = b1
        if step <= _R_ABSTOL:
            break
    out = np.empty(2)
    out[0] = a0
    out[1] = a1
    return out
