"""Digamma, trigamma and inverse digamma for Numba code.

Numba cannot call scipy.special, so these use the recurrence up to x >= 10 and
then the asymptotic series. The relative error is below 1e-14 for x > 0.
"""

import numpy as np
from numba import njit

_EULER = 0.5772156649015329


@njit(cache=True)
def digamma(x):
    """Digamma function."""
    result = 0.0
    while x < 10.0:
        result -= 1.0 / x
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += np.log(x) - 0.5 * inv
    result -= inv2 * (
        1.0 / 12.0
        - inv2
        * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 * (1.0 / 240.0 - inv2 / 132.0)))
    )
    return result


@njit(cache=True)
def trigamma(x):
    """Trigamma function, the derivative of digamma."""
    result = 0.0
    while x < 10.0:
        result += 1.0 / (x * x)
        x += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += inv * (
        1.0
        + 0.5 * inv
        + inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 / 30.0)))
    )
    return result


@njit(cache=True)
def inv_digamma(y, n_newton=5):
    """Inverse digamma with Newton steps (Minka 2000, appendix C)."""
    if y >= -2.22:
        x = np.exp(y) + 0.5
    else:
        x = -1.0 / (y + _EULER)
    for _ in range(n_newton):
        x -= (digamma(x) - y) / trigamma(x)
    return x
