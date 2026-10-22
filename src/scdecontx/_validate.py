"""Checks of the counts and the labels that run() receives."""

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from ._kernels import all_integral


def get_counts(adata: AnnData, layer: str | None, name: str = "adata"):
    """Return the count matrix and a name for it in messages."""
    if layer is None:
        return adata.X, f"{name}.X"
    if layer not in adata.layers:
        available = ", ".join(map(repr, adata.layers.keys())) or "none"
        raise KeyError(
            f"Layer {layer!r} not found in {name}.layers (available: {available})."
        )
    return adata.layers[layer], f"{name}.layers[{layer!r}]"


def to_csr(X) -> csr_matrix:
    """Return X as a canonical CSR matrix. Do not change the matrix of the caller."""
    if issparse(X):
        X = csr_matrix(X)
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()
        return X
    return csr_matrix(np.asarray(X))


def check_counts(adata: AnnData, X: csr_matrix, source: str, *, cells: bool = True):
    """Raise for invalid counts. Warn about input that gives unreliable results.

    Set cells to False for empty droplets: they are allowed to be few and empty.
    """
    data = X.data
    if data.size and data.min() < 0:
        raise ValueError(f"{source} contains negative values.")
    if not np.isfinite(data).all():
        raise ValueError(f"{source} contains NaN or infinite values.")
    # uns['log1p'] stays after raw counts go back into .X, so raise only when
    # the values are also not whole numbers.
    if data.size and not all_integral(data):
        if "log1p" in adata.uns:
            raise ValueError(
                f"{source} has non-integer values and adata.uns['log1p'] is set, so it "
                "appears to be log-transformed. DecontX needs raw counts: pass the raw "
                "layer (for example layer='counts') or run DecontX before normalizing."
            )
        warnings.warn(
            f"{source} contains non-integer values. DecontX expects raw counts.",
            stacklevel=3,
        )
    if not cells:
        return
    # Negative values were rejected above, so a zero total means no counts.
    n_empty = int((np.asarray(X.sum(axis=1)).ravel() == 0).sum())
    if n_empty:
        warnings.warn(
            f"{n_empty} cell(s) have zero total counts in {source}. They get NaN "
            "contamination and do not affect the fit. Filter them out first.",
            stacklevel=3,
        )
    if X.shape[0] < 10:
        warnings.warn(
            "Fewer than 10 cells. The estimates are unreliable.", stacklevel=3
        )
    if X.shape[1] < 100:
        warnings.warn(
            "Fewer than 100 genes. The estimates are unreliable.", stacklevel=3
        )


def labels(adata: AnnData, key: str, what: str) -> np.ndarray:
    """Return obs[key] as an object array. Raise if it is missing or has NaN."""
    if key not in adata.obs:
        raise KeyError(f"{what} key {key!r} not found in adata.obs.")
    values = adata.obs[key]
    n_missing = int(values.isna().sum())
    if n_missing:
        raise ValueError(
            f"{what} key {key!r} has {n_missing} missing value(s). Assign every cell "
            "or remove the cells first."
        )
    values = values.to_numpy(dtype=object)
    # Labels are compared as strings, so distinct labels with the same text
    # (1 and "1") would merge into one cluster or batch.
    by_text = {}
    for u in pd.unique(values):
        by_text.setdefault(str(u), []).append(u)
    clashes = [group for group in by_text.values() if len(group) > 1]
    if clashes:
        examples = "; ".join(" and ".join(map(repr, g)) for g in clashes[:3])
        raise ValueError(
            f"{what} key {key!r} has distinct labels with the same text ({examples}). "
            "Convert the column to one type first."
        )
    return values


def codes(values: np.ndarray, batch_name=None) -> tuple[np.ndarray, int]:
    """Return 0-based cluster codes in sorted label order and the number of clusters."""
    uniques, z = np.unique(values.astype(str), return_inverse=True)
    if len(uniques) < 2:
        where = "" if batch_name is None else f" in batch {batch_name!r}"
        raise ValueError(
            f"DecontX needs at least 2 clusters{where}; found {len(uniques)}."
        )
    return z.astype(np.int64), len(uniques)
