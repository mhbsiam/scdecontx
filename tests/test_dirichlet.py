import numpy as np
import pytest
from _data import FIXTURES
from scipy.special import digamma as sp_digamma
from scipy.special import polygamma

from scdecontx._dirichlet import fit_r
from scdecontx._special import digamma, inv_digamma, trigamma

START = np.array([10.0, 10.0])


def test_special_functions_match_scipy():
    xs = np.logspace(-3, 4, 500)
    dg = np.array([digamma(x) for x in xs])
    tg = np.array([trigamma(x) for x in xs])
    np.testing.assert_allclose(dg, sp_digamma(xs), rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(tg, polygamma(1, xs), rtol=1e-11)
    ys = sp_digamma(np.logspace(-2, 3, 50))
    np.testing.assert_allclose([digamma(inv_digamma(y)) for y in ys], ys, atol=1e-12)


@pytest.mark.parametrize("key", ["", "_with_ones"])
def test_fit_r_matches_mcmcprecision(key):
    """R: MCMCprecision::fit_dirichlet(cbind(p, 1 - p))$alpha with the defaults."""
    ref = np.load(FIXTURES / "dirichlet_r.npz")
    p = ref["p" + key]
    np.testing.assert_allclose(fit_r(p, START), ref["alpha" + key], rtol=1e-13)


def test_degenerate_input_keeps_delta():
    """No spread in the proportions gives no estimate: keep the previous delta."""
    assert np.array_equal(fit_r(np.full(50, 0.9), START), START)
