"""The soup simulator in tests/_soup.py, and how DecontX does on it."""

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from _soup import SCENARIOS, oracle, score, simulate_soup

import scdecontx as dx

SMALL = {"sizes": (600, 300, 150, 80, 40), "n_genes": 1000}


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_truth_adds_up(name):
    s = simulate_soup(0, **{**SMALL, **SCENARIOS[name]})
    soup = (s.X - s.native).toarray()
    total = np.asarray(s.X.sum(axis=1)).ravel()
    assert soup.min() >= 0
    np.testing.assert_allclose(soup.sum(axis=1) / total, s.contamination)
    assert np.all(s.foreign <= s.contamination + 1e-12)
    np.testing.assert_allclose(s.soup_share.sum(), 1.0)
    if not SCENARIOS[name].get("own_soup", True):
        np.testing.assert_allclose(s.foreign, s.contamination)


def test_droplets_follow_the_soup():
    s = simulate_soup(0, **SMALL)
    droplets = np.asarray(s.droplets.sum(axis=0)).ravel()
    soup = np.asarray((s.X - s.native).sum(axis=0)).ravel()
    assert np.corrcoef(droplets, soup)[0, 1] > 0.95


def test_score_of_perfect_and_of_no_estimate():
    s = simulate_soup(0, **SMALL)
    perfect = score(s, s.contamination, s.native)
    assert perfect["mae_total"] == 0
    assert perfect["marker_leak"] == 0
    assert perfect["over_removal"] == 0
    assert perfect["counts_rel_error"] == 0
    none = score(s, np.zeros(len(s.z)), s.X)
    assert none["marker_leak"] == 1
    assert none["over_removal"] == 0
    np.testing.assert_allclose(none["mae_total"], s.contamination.mean())


def _run(s, background=False, **kw):
    adata = ad.AnnData(s.X)
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"g{i}" for i in range(adata.n_vars)]
    adata.obs["cluster"] = pd.Categorical(s.z.astype(str))
    bg = None
    if background:
        bg = ad.AnnData(s.droplets)
        bg.obs_names = [f"d{i}" for i in range(bg.n_obs)]
        bg.var_names = adata.var_names
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dx.run(adata, "cluster", background=bg, **kw)
    return score(
        s,
        adata.obs["scdecontx_contamination"].to_numpy(),
        adata.layers["scdecontx_counts"],
    )


@pytest.fixture(scope="module")
def own_soup():
    return simulate_soup(1, **SMALL, **SCENARIOS["S1_own_soup"])


def test_model_scenario_is_accurate():
    s = simulate_soup(1, delta=(1.0, 10.0), **SMALL, **SCENARIOS["S0_model"])
    assert _run(s)["mae_total"] < 0.01


def test_foreign_contamination_survives_own_soup(own_soup):
    """The part the model can see is still estimated well."""
    assert _run(own_soup)["mae_foreign"] < 0.05


def test_background_recovers_own_soup(own_soup):
    """Soup of the cell's own type looks native; empty droplets reveal it."""
    plain = _run(own_soup)["mae_total"]
    with_bg = _run(own_soup, background=True)["mae_total"]
    assert with_bg < 0.5 * plain


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_oracle_is_the_floor(name):
    """The posterior under the true parameters estimates every scenario well."""
    s = simulate_soup(0, **{**SMALL, **SCENARIOS[name]})
    m = score(s, *oracle(s))
    assert m["mae_total"] < 0.02
    assert m["self_consistency"] < 1e-12


def test_background_is_near_the_oracle_on_realistic_data():
    """With empty droplets the counts are close to the best possible."""
    s = simulate_soup(1, **SMALL, **SCENARIOS["S6_realistic"])
    floor = score(s, *oracle(s))["counts_rel_error"]
    assert _run(s, background=True)["counts_rel_error"] < 1.25 * floor


def test_no_drift_with_a_mildly_different_soup():
    """S7 at 3% contamination, as in the human-mouse mixture.

    Run to convergence, the EM moves the mismatch between soup and cells into
    the native profiles; celda's stop rule (tol 1e-3) ends it before that.
    """
    s = simulate_soup(1, delta=(1.0, 30.0), **SCENARIOS["S7_drift"])
    assert _run(s)["marker_leak"] < 0.02
