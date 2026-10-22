"""Empty droplets that fix the contamination profile, as in celda."""

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from ._validate import check_counts, get_counts, labels, to_csr


class Background(NamedTuple):
    """Empty droplets aligned with adata."""

    droplets: csr_matrix  # droplets x genes, in the gene order of adata
    batch: np.ndarray | None  # batch of each droplet, None without batches
    n_dropped: int  # droplets removed because they are also cells


def prepare(
    adata, background, batch_key, background_batch_key, layer, batches
) -> Background | None:
    """Align the droplets with adata and check them with the rules of celda.

    Return None without background.
    """
    if background is None:
        if background_batch_key is not None:
            raise ValueError("background_batch_key was given without background.")
        return None
    if not isinstance(background, AnnData):
        raise TypeError(
            "background must be an AnnData of raw empty droplets (droplets x genes)."
        )
    if batch_key is None and background_batch_key is not None:
        raise ValueError(
            "background_batch_key was given but batch_key was not. Without cell batches "
            "the droplets are not batched either."
        )
    if batch_key is not None and background_batch_key is None:
        raise ValueError(
            "batch_key and background were given, so background_batch_key is required "
            "to assign each droplet to a batch."
        )

    # Droplets may lack the layer; then their .X holds the raw counts.
    layer = layer if layer in background.layers else None
    X, source = get_counts(background, layer, name="background")
    X = to_csr(X)
    check_counts(background, X, source, cells=False)

    if not background.var_names.is_unique:
        raise ValueError(
            "background.var_names must be unique to match genes with adata."
        )
    missing = ~adata.var_names.isin(background.var_names)
    if missing.any():
        examples = ", ".join(map(str, adata.var_names[missing][:5]))
        raise ValueError(
            f"{int(missing.sum())} gene(s) of adata are missing from background "
            f"(for example {examples}). Use the same features for cells and droplets."
        )
    gene_idx = background.var_names.get_indexer(adata.var_names)

    keep = ~background.obs_names.isin(adata.obs_names)
    n_dropped = int((~keep).sum())
    if n_dropped:
        warnings.warn(
            f"{n_dropped} background droplet(s) have the barcode of a cell in adata. "
            "They were removed from the background.",
            stacklevel=3,
        )
    if not keep.any():
        raise ValueError("No background droplets remain after removing the cells.")
    if not keep.all():
        X = X[np.flatnonzero(keep)]
    if X.shape[1] != len(gene_idx) or not np.array_equal(
        gene_idx, np.arange(len(gene_idx))
    ):
        X = X[:, gene_idx].tocsr()

    droplet_batch = None
    if background_batch_key is not None:
        droplet_batch = labels(background, background_batch_key, "Background batch")[
            keep
        ]
        matched = pd.Series(droplet_batch).isin(batches).to_numpy()
        if not matched.all():
            unknown = ", ".join(map(repr, pd.unique(droplet_batch[~matched])[:5]))
            warnings.warn(
                f"{int((~matched).sum())} background droplet(s) have a batch that no "
                f"cell has (for example {unknown}). They were removed from the "
                "background.",
                stacklevel=3,
            )
            X = X[np.flatnonzero(matched)]
            droplet_batch = droplet_batch[matched]
        totals = np.asarray(X.sum(axis=1)).ravel()
        empty = [b for b in batches if totals[droplet_batch == b].sum() <= 0]
        if empty:
            raise ValueError(
                "No background counts for batch(es) "
                + ", ".join(map(repr, empty))
                + f". background.obs[{background_batch_key!r}] must use the labels of "
                f"adata.obs[{batch_key!r}]."
            )
    elif X.sum() <= 0:
        raise ValueError("The background contains no counts.")
    return Background(X, droplet_batch, n_dropped)


def profile(X, droplet_batch, batch) -> np.ndarray:
    """Return the fixed contamination profile of one batch: the pooled droplets."""
    rows = X if droplet_batch is None else X[np.flatnonzero(droplet_batch == batch)]
    eta = np.asarray(rows.sum(axis=0), dtype=np.float64).ravel() + 1e-20
    return eta / eta.sum()
