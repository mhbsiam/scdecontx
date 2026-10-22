"""Numba kernels for the DecontX EM on a cells x genes CSR matrix.

Notation: theta[j] is the native proportion of cell j, z[j] its cluster
(0-based), phi[k] the native gene profile of cluster k and eta[k] the
contamination profile of cluster k. nc holds the native counts at the CSR
nonzeros, native[j] their sum for each cell, and acc[k] their sum for each
cluster and gene.

Each prange loop writes only to the slice of one cell, one row of acc or a
buffer, or one gene, in a fixed order. The results do not depend on the
thread count.
"""

import numpy as np
from numba import njit, prange

PSEUDOCOUNT = 1e-20


@njit(inline="always")
def _row(indptr, j):
    """Return the range of row j's nonzeros in data and indices."""
    return indptr[j], indptr[j + 1]


@njit(cache=True)
def all_integral(data):
    """Return True if all values are whole numbers."""
    for i in range(len(data)):
        if data[i] != np.floor(data[i]):
            return False
    return True


@njit(parallel=True, cache=True)
def estep(data, indices, indptr, z, theta, phi, eta, nc, native):
    """Calculate the native counts nc and their cell sums native.

    The native probability of a count is (phi + pc)(theta + pc), and the
    contamination probability is (eta + pc)(1 - theta + pc), as in celda.
    """
    pc = PSEUDOCOUNT
    for j in prange(len(theta)):
        lo, hi = _row(indptr, j)
        genes, counts, out = indices[lo:hi], data[lo:hi], nc[lo:hi]
        k, t = z[j], theta[j]
        s = 0.0
        for n in range(hi - lo):
            g = genes[n]
            p_native = (phi[k, g] + pc) * (t + pc)
            p_contam = (eta[k, g] + pc) * (1.0 - t + pc)
            v = counts[n] * (p_native / (p_native + p_contam))
            out[n] = v
            s += v
        native[j] = s


@njit(parallel=True, cache=True)
def accumulate(values, indices, indptr, cell_order, units, buf_starts, bufs, acc):
    """Sum values (at the CSR nonzeros) into acc, one row for each cluster.

    Each unit (cluster k, lo, hi, buffer b) sums the cells cell_order[lo:hi]
    of cluster k. The first unit of a cluster has b = -1 and writes acc[k].
    The others write bufs[b], which are added to acc[k] in unit order. Units
    of similar size keep the threads busy when one cluster holds most cells.
    """
    for u in prange(units.shape[0]):
        b = units[u, 3]
        row = acc[units[u, 0]] if b < 0 else bufs[b]
        row[:] = 0.0
        for ci in range(units[u, 1], units[u, 2]):
            lo, hi = _row(indptr, cell_order[ci])
            genes, vals = indices[lo:hi], values[lo:hi]
            for n in range(hi - lo):
                row[genes[n]] += vals[n]
    if bufs.shape[0] == 0:
        return
    n_clusters, n_genes = acc.shape
    for g in prange(n_genes):
        for k in range(n_clusters):
            s = acc[k, g]
            for b in range(buf_starts[k], buf_starts[k + 1]):
                s += bufs[b, g]
            acc[k, g] = s


@njit(parallel=True, cache=True)
def profiles(acc, phi, eta, estimate_eta):
    """Normalize acc into phi and, if estimate_eta, the other clusters into eta.

    A row without counts gets a uniform profile. Each gene total and each row
    is summed in cluster or gene order, so the result is thread-independent.
    """
    n_clusters, n_genes = acc.shape
    gene_tot = np.zeros(n_genes)
    for g in prange(n_genes):
        t = 0.0
        for k in range(n_clusters):
            t += acc[k, g]
        gene_tot[g] = t
    for k in prange(n_clusters):
        s = 0.0
        for g in range(n_genes):
            s += acc[k, g]
        for g in range(n_genes):
            phi[k, g] = acc[k, g] / s if s > 0.0 else 1.0 / n_genes
        if estimate_eta:
            s = 0.0
            for g in range(n_genes):
                s += gene_tot[g] - acc[k, g]
            for g in range(n_genes):
                eta[k, g] = (gene_tot[g] - acc[k, g]) / s if s > 0.0 else 1.0 / n_genes


@njit(parallel=True, cache=True)
def weighted(data, indptr, theta, out):
    """Write data * theta[cell] into out at each CSR nonzero."""
    for j in prange(len(theta)):
        lo, hi = _row(indptr, j)
        counts, row = data[lo:hi], out[lo:hi]
        t = theta[j]
        for n in range(hi - lo):
            row[n] = counts[n] * t
