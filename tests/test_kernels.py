"""The Numba EM agrees with the NumPy oracle in tests/_oracle.py."""

import numpy as np
import pytest
from _data import simulate
from _oracle import em
from scipy.sparse import csr_matrix

from scdecontx import _em
from scdecontx import _kernels as K
from scdecontx._em import fit_batch, plan_units


def _fit(X, z, theta0, steps):
    return fit_batch(
        X,
        z,
        z.max() + 1,
        delta=(10.0, 10.0),
        estimate_delta=True,
        tol=0.0,
        max_iter=steps,
        rng=None,
        _theta0=theta0,
    )


@pytest.mark.parametrize("steps", [1, 3, 25])
def test_em_matches_oracle(steps):
    X, z, _ = simulate(0, n_cells=300, n_genes=150, n_clusters=3)
    theta0 = np.random.default_rng(1).beta(10, 10, size=X.shape[0])
    res = _fit(X, z, theta0.copy(), steps)
    ref = em(X, z, theta0.copy(), tol=0.0, max_iter=steps)
    np.testing.assert_allclose(res.theta, ref["theta"], atol=1e-12)
    np.testing.assert_allclose(res.contamination, ref["contamination_iter"], atol=1e-12)
    np.testing.assert_allclose(res.counts.data, ref["native"], atol=1e-10)
    np.testing.assert_allclose(res.delta, ref["delta"], rtol=1e-12)


def test_singleton_cluster_stays_finite():
    X, z, _ = simulate(3, n_cells=200, n_genes=100, n_clusters=3)
    z = z.copy()
    z[0] = 3
    res = fit_batch(
        X,
        z,
        4,
        delta=(10.0, 10.0),
        estimate_delta=True,
        tol=1e-3,
        max_iter=500,
        rng=np.random.default_rng(0),
    )
    assert np.isfinite(res.contamination).all()
    assert res.converged


@pytest.mark.parametrize("target", [1e9, 500, 37])
def test_accumulate_matches_cluster_sums(target):
    """Clusters split into units sum to the same profiles as unsplit ones."""
    X, z, _ = simulate(5, n_cells=300, n_genes=150, n_clusters=3)
    z = z.copy()
    z[:200] = 0  # one large cluster, split into many units
    z[200] = 3  # a singleton
    X = X.tolil()
    X[201, :] = 0  # a cell without counts
    X = csr_matrix(X)
    values = X.data * np.random.default_rng(0).random(X.nnz)
    order = np.argsort(z, kind="stable")
    starts = np.zeros(5, dtype=np.int64)
    np.cumsum(np.bincount(z, minlength=4), out=starts[1:])
    units, buf_starts = plan_units(X.indptr, order, starts, target)
    assert set(units[:, 0]) == {0, 1, 2, 3}
    acc = np.full((4, X.shape[1]), np.nan)
    bufs = np.full((buf_starts[-1], X.shape[1]), np.nan)
    K.accumulate(values, X.indices, X.indptr, order, units, buf_starts, bufs, acc)
    V = csr_matrix((values, X.indices, X.indptr), shape=X.shape).toarray()
    expected = np.stack([V[z == k].sum(axis=0) for k in range(4)])
    np.testing.assert_allclose(acc, expected, rtol=1e-12, atol=1e-12)


def test_split_units_match_oracle(monkeypatch):
    monkeypatch.setattr(_em, "MIN_UNIT_NNZ", 1)
    monkeypatch.setattr(_em, "UNITS", 50)
    test_em_matches_oracle(25)
