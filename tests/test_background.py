import anndata as ad
import numpy as np
import pytest
from _data import make_adata
from scipy.sparse import csr_matrix

import scdecontx as dx
from scdecontx._em import fit_batch
from scdecontx._validate import codes


def _droplets(adata, n=60, seed=0, batch=None):
    rng = np.random.default_rng(seed)
    profile = np.asarray(adata.X.sum(axis=0)).ravel()
    X = rng.multinomial(40, profile / profile.sum(), size=n).astype(float)
    bg = ad.AnnData(csr_matrix(X))
    bg.obs_names = [f"empty{i}" for i in range(n)]
    bg.var_names = adata.var_names
    if batch is not None:
        bg.obs["batch"] = batch
    return bg


def test_background_fixes_eta_to_the_droplet_profile():
    adata, _ = make_adata()
    bg = _droplets(adata)
    dx.run(adata, "leiden", background=bg, random_state=3)
    z, k = codes(adata.obs["leiden"].to_numpy(dtype=object))
    eta = np.asarray(bg.X.sum(axis=0)).ravel() + 1e-20
    res = fit_batch(
        adata.X,
        z,
        k,
        tol=1e-3,
        max_iter=500,
        delta=(10.0, 10.0),
        estimate_delta=True,
        rng=np.random.default_rng(np.random.SeedSequence(3).spawn(1)[0]),
        eta_fixed=np.tile(eta / eta.sum(), (k, 1)),
    )
    assert np.array_equal(adata.obs["scdecontx_contamination"], res.contamination)
    assert adata.uns["scdecontx"]["params"]["background"] is True
    assert adata.uns["scdecontx"]["background"]["n_droplets"] == 60


def test_genes_are_matched_by_name():
    adata, _ = make_adata()
    bg = _droplets(adata)
    shuffled = bg[:, bg.var_names[::-1]].copy()
    a = adata.copy()
    dx.run(adata, "leiden", background=bg)
    dx.run(a, "leiden", background=shuffled)
    assert np.array_equal(
        adata.obs["scdecontx_contamination"], a.obs["scdecontx_contamination"]
    )
    with pytest.raises(ValueError, match="missing from background"):
        dx.run(adata, "leiden", background=bg[:, 1:].copy())


def test_droplets_that_are_cells_are_removed():
    adata, _ = make_adata()
    bg = _droplets(adata)
    bg.obs_names = list(adata.obs_names[:3]) + list(bg.obs_names[3:])
    with pytest.warns(UserWarning, match="3 background droplet"):
        dx.run(adata, "leiden", background=bg)
    assert adata.uns["scdecontx"]["background"]["n_dropped_overlap"] == 3
    assert adata.uns["scdecontx"]["background"]["n_droplets"] == 57


def test_batch_rules_follow_r():
    adata, _ = make_adata()
    adata.obs["batch"] = np.where(np.arange(adata.n_obs) % 2, "A", "B")
    bg = _droplets(adata, batch=np.where(np.arange(60) % 2, "A", "B"))
    with pytest.raises(ValueError, match="background_batch_key is required"):
        dx.run(adata, "leiden", batch_key="batch", background=bg)
    with pytest.raises(ValueError, match="batch_key was not"):
        dx.run(adata, "leiden", background=bg, background_batch_key="batch")
    with pytest.raises(ValueError, match="without background"):
        dx.run(adata, "leiden", background_batch_key="batch")
    bg.obs["batch"] = "A"
    with pytest.raises(ValueError, match="No background counts for batch"):
        dx.run(
            adata,
            "leiden",
            batch_key="batch",
            background=bg,
            background_batch_key="batch",
        )


def test_each_batch_uses_its_own_droplets():
    adata, _ = make_adata()
    adata.obs["batch"] = np.where(np.arange(adata.n_obs) % 2, "A", "B")
    bg = _droplets(adata, batch=np.where(np.arange(60) % 2, "A", "B"))
    dx.run(
        adata, "leiden", batch_key="batch", background=bg, background_batch_key="batch"
    )
    swapped = bg.copy()
    swapped.obs["batch"] = np.where(bg.obs["batch"] == "A", "B", "A")
    other = adata.copy()
    dx.run(
        other,
        "leiden",
        batch_key="batch",
        background=swapped,
        background_batch_key="batch",
    )
    assert not np.array_equal(
        adata.obs["scdecontx_contamination"], other.obs["scdecontx_contamination"]
    )


def test_droplets_of_unknown_batches_are_removed():
    adata, _ = make_adata()
    adata.obs["batch"] = np.where(np.arange(adata.n_obs) % 2, "A", "B")
    bg = _droplets(adata, n=90, batch=np.repeat(["A", "B", "C"], 30))
    kw = {"batch_key": "batch", "background_batch_key": "batch", "random_state": 3}
    with pytest.warns(UserWarning, match="30 background droplet.*'C'"):
        dx.run(adata, "leiden", background=bg, **kw)
    assert adata.uns["scdecontx"]["background"]["n_droplets"] == 60
    known = adata.copy()
    dx.run(known, "leiden", background=bg[:60].copy(), **kw)
    assert np.array_equal(
        adata.obs["scdecontx_contamination"], known.obs["scdecontx_contamination"]
    )


def test_empty_background_raises():
    adata, _ = make_adata()
    bg = _droplets(adata)
    bg.X = csr_matrix(bg.shape)
    with pytest.raises(ValueError, match="no counts"):
        dx.run(adata, "leiden", background=bg)
    with pytest.raises(TypeError):
        dx.run(adata, "leiden", background=bg.X)
