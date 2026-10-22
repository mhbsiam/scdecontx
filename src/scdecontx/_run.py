"""The public entry point."""

import logging
import warnings
from typing import Literal

import numpy as np
from anndata import AnnData
from scipy.sparse import csr_matrix

from . import __version__, _background
from ._em import fit_batch
from ._validate import check_counts, codes, get_counts, labels, to_csr

logger = logging.getLogger("scdecontx")


def run(
    adata: AnnData,
    groupby: str,
    *,
    batch_key: str | None = None,
    layer: str | None = None,
    background: AnnData | None = None,
    background_batch_key: str | None = None,
    delta: tuple[float, float] = (10.0, 10.0),
    estimate_delta: bool = True,
    max_iter: int = 500,
    tol: float = 1e-3,
    random_state: int = 0,
    dtype: Literal["float64", "float32"] = "float64",
    key_added: str = "scdecontx",
    copy: bool = False,
) -> AnnData | None:
    """Estimate and remove ambient RNA contamination with DecontX.

    DecontX models the counts of each cell as a mixture of its own (native)
    expression and contamination from the other cell populations. It needs
    raw counts and a clustering of broad cell types. This is celda's
    ``decontX``: from the same initial values it gives the same estimates.

    Parameters
    ----------
    adata
        Cells x genes. Raw counts are in ``.X`` or in ``layers[layer]``.
    groupby
        Column of ``adata.obs`` with the cluster of each cell.
    batch_key
        Column of ``adata.obs`` with batches. Each batch is fitted separately.
    layer
        Layer with the raw counts. Default: ``.X``.
    background
        Raw empty droplets (droplets x genes). Their pooled profile replaces
        the estimated contamination profile, as in R ``decontX(background=)``.
        Genes are matched by ``var_names``. Droplets with a barcode of a cell
        are removed.
    background_batch_key
        Column of ``background.obs`` with the batch of each droplet. Required
        when ``batch_key`` and ``background`` are given. Droplets of a batch
        without cells are removed.
    delta
        Beta prior of the native proportion, and the start of its fit.
    estimate_delta
        Fit ``delta`` in each iteration.
    max_iter
        Maximum EM iterations, as celda's ``maxIter``.
    tol
        Stop when no native proportion changes by more than ``tol`` in one
        iteration, as celda's ``convergence``.
    random_state
        Seed of the initial native proportions. Each batch gets its own seed.
    dtype
        Precision of the counts and the output layer. ``"float32"`` halves the
        memory of both.
    key_added
        Prefix of the output keys.
    copy
        Return a copy instead of writing to ``adata``.

    Returns
    -------
    A copy of ``adata`` if ``copy`` is true, otherwise ``None``. Sets:

    ``obs[f"{key_added}_contamination"]``
        Contamination fraction of each cell. ``NaN`` for cells without counts.
    ``layers[f"{key_added}_counts"]``
        Decontaminated counts (fractional), with the sparsity of the input.
    ``uns[key_added]``
        Parameters, the fitted ``delta`` and the convergence of each batch.
    """
    tol, max_iter = float(tol), int(max_iter)
    if not (np.isfinite(tol) and tol > 0) or max_iter < 1:
        raise ValueError("tol must be finite and positive and max_iter at least 1.")
    delta = tuple(float(d) for d in delta)
    if len(delta) != 2 or not all(np.isfinite(d) and d > 0 for d in delta):
        raise ValueError("delta must be two finite positive numbers.")
    if dtype not in ("float64", "float32"):
        raise ValueError(f"dtype must be 'float64' or 'float32', not {dtype!r}.")

    if copy:
        adata = adata.copy()
    X, source = get_counts(adata, layer)
    X = to_csr(X)
    check_counts(adata, X, source)
    clusters = labels(adata, groupby, "Cluster")

    if batch_key is None:
        batch_of_cell = None
        batches = [None]
    else:
        batch_of_cell = labels(adata, batch_key, "Batch")
        batches = list(dict.fromkeys(batch_of_cell))
    bg = _background.prepare(
        adata, background, batch_key, background_batch_key, layer, batches
    )

    seeds = np.random.SeedSequence(random_state).spawn(len(batches))
    if batch_key is not None:
        # Filled batch by batch; without batches the one fit's arrays are used.
        contamination = np.empty(adata.n_obs)
        out_data = np.empty(X.nnz, dtype=dtype)
    fits = []
    for batch, seed in zip(batches, seeds, strict=True):
        rows = (
            np.arange(adata.n_obs)
            if batch is None
            else np.flatnonzero(batch_of_cell == batch)
        )
        Xb = X if batch is None else X[rows]
        z, n_clusters = codes(clusters[rows], batch)
        eta = None
        if bg is not None:
            eta = _background.profile(bg.droplets, bg.batch, batch)
        res = fit_batch(
            Xb,
            z,
            n_clusters,
            delta=delta,
            estimate_delta=estimate_delta,
            tol=tol,
            max_iter=max_iter,
            rng=np.random.default_rng(seed),
            dtype=np.dtype(dtype),
            eta_fixed=eta,
        )
        name = "all" if batch is None else str(batch)
        if batch is None:
            contamination = res.contamination
            out_data = res.counts.data
        else:
            contamination[rows] = res.contamination
            # Rows are ascending, so the batch's nonzeros keep their order in X.
            in_batch = np.zeros(adata.n_obs, dtype=bool)
            in_batch[rows] = True
            out_data[np.repeat(in_batch, np.diff(X.indptr))] = res.counts.data
        fits.append((name, res))
        if res.n_iter == 0:
            warnings.warn(
                f"Batch {name!r} has no counts. Its cells get NaN contamination and "
                "were not fitted.",
                stacklevel=2,
            )
            continue
        logger.info(
            "batch %s: %d iterations, converged %s, mean contamination %.3f",
            name,
            res.n_iter,
            res.converged,
            np.nanmean(res.contamination),
        )

    stalled = [f"{b} (change {r.final_change:.2e})" for b, r in fits if not r.converged]
    if stalled:
        warnings.warn(
            f"DecontX did not converge within max_iter={max_iter} for: "
            f"{', '.join(stalled)}. "
            "Raise max_iter or tol.",
            stacklevel=2,
        )

    adata.obs[f"{key_added}_contamination"] = contamination
    adata.layers[f"{key_added}_counts"] = csr_matrix(
        (out_data, X.indices.copy(), X.indptr.copy()), shape=X.shape
    )
    adata.uns[key_added] = _summary(
        fits,
        bg,
        groupby=groupby,
        batch_key=batch_key,
        layer=layer,
        delta=list(delta),
        estimate_delta=estimate_delta,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state,
        dtype=dtype,
        background=bg is not None,
        background_batch_key=background_batch_key,
    )
    return adata if copy else None


def _summary(fits, bg, **params) -> dict:
    """The uns entry: the parameters, the fit of each batch and the background.

    h5py cannot store None, so absent parameters become "". Batches are
    parallel lists, not a dict keyed by batch, because h5py splits keys at "/".
    """
    return {
        "params": {k: "" if v is None else v for k, v in params.items()},
        "fit": {
            "batch": [b for b, _ in fits],
            "delta_native": [float(r.delta[0]) for _, r in fits],
            "delta_contamination": [float(r.delta[1]) for _, r in fits],
            "n_iter": [int(r.n_iter) for _, r in fits],
            "converged": [bool(r.converged) for _, r in fits],
            "final_change": [float(r.final_change) for _, r in fits],
        },
        "background": {
            "n_droplets": 0 if bg is None else int(bg.droplets.shape[0]),
            "n_dropped_overlap": 0 if bg is None else int(bg.n_dropped),
        },
        "version": __version__,
    }
