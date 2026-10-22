import numpy as np
import pytest
from _data import make_adata
from scipy.sparse import csr_matrix

import scdecontx as dx
from scdecontx._em import fit_batch
from scdecontx._validate import codes


def _batched(names=("A", "B")):
    adata, _ = make_adata(n_cells=400)
    adata.obs["batch"] = np.array(names)[np.arange(adata.n_obs) % len(names)]
    return adata


def test_each_batch_equals_a_separate_fit_with_its_seed():
    adata = _batched()
    dx.run(adata, "leiden", batch_key="batch", random_state=7)
    seeds = np.random.SeedSequence(7).spawn(2)
    for name, seed in zip(["A", "B"], seeds, strict=True):
        sub = adata[adata.obs["batch"] == name].copy()
        z, k = codes(sub.obs["leiden"].to_numpy(dtype=object))
        res = fit_batch(
            sub.X,
            z,
            k,
            tol=1e-3,
            max_iter=500,
            delta=(10.0, 10.0),
            estimate_delta=True,
            rng=np.random.default_rng(seed),
        )
        assert np.array_equal(
            sub.obs["scdecontx_contamination"].to_numpy(), res.contamination
        )
        assert np.array_equal(sub.layers["scdecontx_counts"].data, res.counts.data)


def test_batches_get_different_seeds():
    adata = _batched()
    adata.obs["batch"] = np.where(np.arange(adata.n_obs) < 200, "A", "B")
    adata.obs.loc[adata.obs_names[200:], "leiden"] = (
        adata.obs["leiden"].iloc[:200].to_numpy()
    )
    X = adata.X.toarray()
    X[200:] = X[:200]
    adata.X = csr_matrix(X)
    dx.run(adata, "leiden", batch_key="batch")
    c = adata.obs["scdecontx_contamination"].to_numpy()
    assert not np.array_equal(c[:200], c[200:])


def test_fit_is_reported_for_each_batch():
    adata = _batched(("x/1", "all", "y"))
    dx.run(adata, "leiden", batch_key="batch")
    fit = adata.uns["scdecontx"]["fit"]
    assert fit["batch"] == ["x/1", "all", "y"]
    assert len(fit["n_iter"]) == len(fit["delta_native"]) == 3
    assert all(fit["converged"])


def test_batch_errors():
    adata = _batched()
    adata.obs.loc[adata.obs_names[0], "batch"] = None
    with pytest.raises(ValueError, match="missing value"):
        dx.run(adata, "leiden", batch_key="batch")
    adata = _batched()
    adata.obs["leiden"] = adata.obs["leiden"].astype(str)
    adata.obs.loc[adata.obs["batch"] == "B", "leiden"] = "0"
    with pytest.raises(ValueError, match="in batch 'B'"):
        dx.run(adata, "leiden", batch_key="batch")
    with pytest.raises(KeyError, match="not found"):
        dx.run(adata, "leiden", batch_key="missing")


@pytest.mark.filterwarnings("ignore:200 cell:UserWarning")
def test_empty_batch_gets_nan_and_does_not_affect_others():
    adata = _batched()
    X = adata.X.toarray()
    X[(adata.obs["batch"] == "A").to_numpy()] = 0
    adata.X = csr_matrix(X)
    with pytest.warns(UserWarning, match="Batch 'A' has no counts"):
        dx.run(adata, "leiden", batch_key="batch", random_state=7)
    a = adata[adata.obs["batch"] == "A"]
    assert np.isnan(a.obs["scdecontx_contamination"]).all()
    assert not a.layers["scdecontx_counts"].data.any()
    fit = adata.uns["scdecontx"]["fit"]
    assert fit["n_iter"][0] == 0 and fit["converged"] == [True, True]

    sub = adata[adata.obs["batch"] == "B"].copy()
    z, k = codes(sub.obs["leiden"].to_numpy(dtype=object))
    res = fit_batch(
        sub.X,
        z,
        k,
        tol=1e-3,
        max_iter=500,
        delta=(10.0, 10.0),
        estimate_delta=True,
        rng=np.random.default_rng(np.random.SeedSequence(7).spawn(2)[1]),
    )
    assert np.array_equal(
        sub.obs["scdecontx_contamination"].to_numpy(), res.contamination
    )


def test_batch_labels_with_the_same_text_are_rejected():
    adata = _batched()
    adata.obs["batch"] = np.array([1, "1"], dtype=object)[np.arange(adata.n_obs) % 2]
    with pytest.raises(ValueError, match="same text"):
        dx.run(adata, "leiden", batch_key="batch")
