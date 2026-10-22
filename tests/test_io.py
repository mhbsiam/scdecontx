import anndata as ad
import numpy as np
import pytest
from _data import make_adata
from scipy.sparse import csr_matrix

import scdecontx as dx


def test_results_survive_h5ad(tmp_path):
    adata, _ = make_adata(n_cells=400)
    adata.obs["batch"] = np.where(np.arange(adata.n_obs) % 2, "a/1", "b")
    X = adata.X.toarray()
    X[3] = 0
    adata.X = csr_matrix(X)
    with pytest.warns(UserWarning, match="zero total counts"):
        dx.run(adata, "leiden", batch_key="batch")
    path = tmp_path / "out.h5ad"
    adata.write_h5ad(path)
    back = ad.read_h5ad(path)
    c = back.obs["scdecontx_contamination"].to_numpy()
    assert np.isnan(c[3])
    assert np.array_equal(
        c, adata.obs["scdecontx_contamination"].to_numpy(), equal_nan=True
    )
    assert list(back.uns["scdecontx"]["fit"]["batch"]) == ["b", "a/1"]
    assert (
        back.layers["scdecontx_counts"] != adata.layers["scdecontx_counts"]
    ).nnz == 0
