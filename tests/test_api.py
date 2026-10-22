import warnings

import numpy as np
import pandas as pd
import pytest
from _data import make_adata
from scipy.sparse import csr_matrix

import scdecontx as dx


def test_public_api_is_one_function():
    assert dx.__all__ == ["run"]
    assert [n for n in dir(dx) if not n.startswith("_")] == ["run"]


def test_outputs():
    adata, _ = make_adata()
    X_before = adata.X.copy()
    assert dx.run(adata, "leiden") is None
    c = adata.obs["scdecontx_contamination"]
    assert c.between(0, 1).all()
    counts = adata.layers["scdecontx_counts"]
    assert counts.shape == adata.shape
    assert counts.dtype == np.float64
    assert np.array_equal(counts.indptr, adata.X.indptr)
    assert (counts.data <= adata.X.data + 1e-12).all()
    assert (adata.X != X_before).nnz == 0
    uns = adata.uns["scdecontx"]
    assert uns["params"]["tol"] == 1e-3
    assert uns["params"]["max_iter"] == 500
    assert "mode" not in uns["params"]
    assert uns["fit"]["batch"] == ["all"]
    assert uns["fit"]["converged"] == [True]
    assert uns["version"] == dx.__version__


def test_tol_and_max_iter_overrides():
    adata, _ = make_adata()
    dx.run(adata, "leiden", tol=1e-4, max_iter=50)
    assert adata.uns["scdecontx"]["params"]["tol"] == 1e-4
    assert adata.uns["scdecontx"]["params"]["max_iter"] == 50


def test_copy_and_key_added():
    adata, _ = make_adata()
    out = dx.run(adata, "leiden", copy=True, key_added="dx")
    assert (
        "dx_contamination" in out.obs and "dx_counts" in out.layers and "dx" in out.uns
    )
    assert "dx_contamination" not in adata.obs


def test_reproducible_and_seeded():
    a, _ = make_adata()
    b = a.copy()
    c = a.copy()
    dx.run(a, "leiden", random_state=1)
    dx.run(b, "leiden", random_state=1)
    dx.run(c, "leiden", random_state=2, tol=1e-3)
    assert np.array_equal(
        a.obs["scdecontx_contamination"], b.obs["scdecontx_contamination"]
    )
    assert not np.array_equal(
        a.obs["scdecontx_contamination"], c.obs["scdecontx_contamination"]
    )


def test_global_rng_is_untouched():
    adata, _ = make_adata()
    np.random.seed(5)
    expected = np.random.rand()
    np.random.seed(5)
    dx.run(adata, "leiden")
    assert np.random.rand() == expected


def test_layer():
    adata, _ = make_adata()
    adata.layers["counts"] = adata.X.copy()
    adata.X = adata.X * 0.5
    ref = adata.copy()
    ref.X = ref.layers["counts"]
    dx.run(adata, "leiden", layer="counts")
    dx.run(ref, "leiden")
    assert np.array_equal(
        adata.obs["scdecontx_contamination"], ref.obs["scdecontx_contamination"]
    )
    assert adata.uns["scdecontx"]["params"]["layer"] == "counts"
    with pytest.raises(KeyError, match="not found in adata.layers"):
        dx.run(adata, "leiden", layer="missing")


def test_dense_input_matches_sparse():
    adata, _ = make_adata()
    dense = adata.copy()
    dense.X = adata.X.toarray()
    dx.run(adata, "leiden")
    dx.run(dense, "leiden")
    assert np.allclose(
        adata.obs["scdecontx_contamination"], dense.obs["scdecontx_contamination"]
    )


def test_float32():
    a, _ = make_adata()
    b = a.copy()
    dx.run(a, "leiden")
    dx.run(b, "leiden", dtype="float32")
    assert b.layers["scdecontx_counts"].dtype == np.float32
    assert np.allclose(
        a.obs["scdecontx_contamination"], b.obs["scdecontx_contamination"], atol=1e-4
    )


def test_cluster_label_types_give_the_same_result():
    a, _ = make_adata()
    b = a.copy()
    b.obs["leiden"] = b.obs["leiden"].astype(int)
    dx.run(a, "leiden")
    dx.run(b, "leiden")
    assert np.array_equal(
        a.obs["scdecontx_contamination"], b.obs["scdecontx_contamination"]
    )


def test_label_errors():
    adata, _ = make_adata()
    with pytest.raises(KeyError, match="not found"):
        dx.run(adata, "missing")
    adata.obs["one"] = "1"
    with pytest.raises(ValueError, match="at least 2 clusters"):
        dx.run(adata, "one")
    adata.obs["gaps"] = pd.Series(
        [None] + ["1"] * (adata.n_obs - 1), index=adata.obs_names, dtype=object
    )
    with pytest.raises(ValueError, match="missing value"):
        dx.run(adata, "gaps")
    # 1 and "1" are distinct labels that would merge as text.
    adata.obs["mixed"] = pd.Series(
        np.array([1, "1", 2, "2"], dtype=object)[np.arange(adata.n_obs) % 4],
        index=adata.obs_names,
        dtype=object,
    )
    with pytest.raises(ValueError, match=r"same text \(1 and '1'; 2 and '2'\)"):
        dx.run(adata, "mixed")


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, -np.inf])
def test_count_errors(value):
    adata, _ = make_adata()
    adata.X = adata.X.copy()
    adata.X.data[0] = value
    match = "negative" if value < 0 else "NaN or infinite"
    with pytest.raises(ValueError, match=match):
        dx.run(adata, "leiden")


def test_nonfinite_background_is_rejected():
    adata, _ = make_adata()
    bg = adata.copy()
    bg.obs_names = [f"empty_{i}" for i in range(bg.n_obs)]
    bg.X = bg.X.copy()
    bg.X.data[0] = np.inf
    with pytest.raises(ValueError, match="background.X contains NaN or infinite"):
        dx.run(adata, "leiden", background=bg)


def test_log_normalized_input_is_rejected():
    adata, _ = make_adata()
    adata.X = csr_matrix(np.log1p(adata.X.toarray()))
    adata.uns["log1p"] = {"base": None}
    with pytest.raises(ValueError, match="log-transformed"):
        dx.run(adata, "leiden")
    del adata.uns["log1p"]
    with pytest.warns(UserWarning, match="non-integer"):
        dx.run(adata, "leiden", tol=1e-3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tol": 0},
        {"tol": np.nan},
        {"tol": np.inf},
        {"max_iter": 0},
        {"delta": (1.0, -1.0)},
        {"delta": (np.nan, 10.0)},
        {"delta": (10.0, np.inf)},
        {"dtype": "float16"},
    ],
)
def test_parameter_errors(kwargs):
    adata, _ = make_adata()
    with pytest.raises(ValueError):
        dx.run(adata, "leiden", **kwargs)


def test_empty_cells_get_nan_and_do_not_affect_others():
    adata, _ = make_adata()
    padded = adata.copy()
    X = padded.X.toarray()
    X[[5, 17]] = 0
    padded.X = csr_matrix(X)
    trimmed = padded[[i for i in range(padded.n_obs) if i not in (5, 17)]].copy()
    with pytest.warns(UserWarning, match="2 cell"):
        dx.run(padded, "leiden", tol=1e-6, max_iter=5000)
    dx.run(trimmed, "leiden", tol=1e-6, max_iter=5000)
    c = padded.obs["scdecontx_contamination"].to_numpy()
    assert np.isnan(c[[5, 17]]).all()
    keep = np.isfinite(c)
    # The empty cells draw their own initial theta, so compare up to tol.
    assert np.allclose(c[keep], trimmed.obs["scdecontx_contamination"], atol=2e-3)


def test_non_convergence_warns():
    adata, _ = make_adata()
    with pytest.warns(UserWarning, match="did not converge"):
        dx.run(adata, "leiden", max_iter=2)
    assert adata.uns["scdecontx"]["fit"]["converged"] == [False]


def test_converged_run_does_not_warn():
    adata, _ = make_adata()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dx.run(adata, "leiden")
