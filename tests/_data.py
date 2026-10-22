"""Test data: celda fixtures and a simulator with known contamination."""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

FIXTURES = Path(__file__).parent / "fixtures"


def read_csr(directory, prefix):
    """Read a genes x cells dgCMatrix written by tests/benchmarks/scripts/io.R.

    The CSC slots of R are the CSR slots of the cells x genes matrix.
    """
    directory = Path(directory)
    n_genes, n_cells = (
        int(v) for v in (directory / f"{prefix}_dim.txt").read_text().split()
    )
    p = np.fromfile(directory / f"{prefix}_p.bin", dtype=np.int32)
    i = np.fromfile(directory / f"{prefix}_i.bin", dtype=np.int32)
    x = np.fromfile(directory / f"{prefix}_x.bin", dtype=np.float64)
    return csr_matrix((x, i, p), shape=(n_cells, n_genes))


def simulate(seed, n_cells=1000, n_genes=500, n_clusters=4, delta=(4.0, 6.0)):
    """Simulate counts as the DecontX model does. Return X, 0-based z and the truth.

    Each cell draws its native counts from the profile of its cluster and its
    contamination from the size-weighted profile of the other clusters.
    delta=(4, 6) gives approximately 40% contamination.
    """
    rng = np.random.default_rng(seed)
    phi = rng.dirichlet(np.full(n_genes, 0.1), size=n_clusters)
    z = rng.integers(n_clusters, size=n_cells)
    weight = np.bincount(z, minlength=n_clusters).astype(float)
    eta = np.empty_like(phi)
    for k in range(n_clusters):
        w = weight.copy()
        w[k] = 0
        eta[k] = w @ phi / w.sum()
    total = rng.integers(500, 1001, size=n_cells)
    n_contam = rng.binomial(total, rng.beta(*delta, size=n_cells))
    X = np.empty((n_cells, n_genes))
    for c in range(n_cells):
        X[c] = rng.multinomial(total[c] - n_contam[c], phi[z[c]])
        X[c] += rng.multinomial(n_contam[c], eta[z[c]])
    return csr_matrix(X), z, n_contam / total


def make_adata(seed=0, n_cells=300, n_genes=200, n_clusters=3, delta=(2.0, 8.0)):
    """Return a small AnnData with clusters in obs['leiden'] and the true contamination."""
    X, z, truth = simulate(seed, n_cells, n_genes, n_clusters, delta)
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell{i}" for i in range(adata.n_obs)]
    adata.var_names = [f"gene{i}" for i in range(adata.n_vars)]
    adata.obs["leiden"] = pd.Categorical(z.astype(str))
    return adata, truth
