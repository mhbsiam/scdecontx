"""scdecontx reproduces celda 1.24.0.

tests/make_celda_fixtures.R ran celda's decontX and captured the initial
theta that it drew. From the same theta, scdecontx must give the same result.
"""

import numpy as np
import pytest
from _data import FIXTURES, read_csr

from scdecontx import _background
from scdecontx._em import fit_batch

TOL = 1e-10


def _fit(name):
    d = FIXTURES / "celda" / name
    X = read_csr(d, "counts")
    z = np.fromfile(d / "z.bin").astype(int) - 1
    n_clusters = z.max() + 1
    eta = None
    if (d / "background_x.bin").exists():
        eta = _background.profile(read_csr(d, "background"), None, None)
    res = fit_batch(
        X,
        z,
        n_clusters,
        tol=1e-3,
        max_iter=500,
        delta=(10.0, 10.0),
        estimate_delta=True,
        rng=None,
        eta_fixed=eta,
        _theta0=np.fromfile(d / "theta0.bin"),
    )
    return d, res


@pytest.mark.parametrize("name", ["low", "high", "background"])
def test_matches_celda(name):
    d, res = _fit(name)
    assert res.n_iter == int((d / "n_iter.txt").read_text())
    np.testing.assert_allclose(
        res.contamination, np.fromfile(d / "contamination.bin"), rtol=0, atol=TOL
    )
    np.testing.assert_allclose(
        res.theta, np.fromfile(d / "theta.bin"), rtol=0, atol=TOL
    )
    np.testing.assert_allclose(res.delta, np.fromfile(d / "delta.bin"), rtol=TOL)
    counts = read_csr(d, "decontx")
    assert abs(res.counts - counts).max() < TOL * max(1.0, counts.max())


def test_fixtures_come_from_celda_1_24():
    assert (FIXTURES / "celda" / "VERSION").read_text().strip() == "celda 1.24.0"
