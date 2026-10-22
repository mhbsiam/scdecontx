"""Accuracy against simulated truth. tests/benchmarks/ measures more."""

import numpy as np
import pytest
from _data import simulate

from scdecontx._em import fit_batch


@pytest.mark.parametrize(
    ("n_genes", "delta", "max_mae"),
    [(500, (4.0, 6.0), 0.08), (500, (1.0, 10.0), 0.03)],
    ids=["40%", "9%"],
)
def test_contamination_is_close_to_truth(n_genes, delta, max_mae):
    X, z, truth = simulate(0, n_cells=800, n_genes=n_genes, delta=delta)
    res = fit_batch(
        X,
        z,
        z.max() + 1,
        delta=(10.0, 10.0),
        estimate_delta=True,
        tol=1e-3,
        max_iter=500,
        rng=np.random.default_rng(0),
    )
    assert res.converged
    assert np.abs(res.contamination - truth).mean() < max_mae
    assert np.corrcoef(res.contamination, truth)[0, 1] > 0.95
