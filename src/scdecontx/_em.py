"""The DecontX EM for one batch."""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from . import _kernels as K
from ._dirichlet import fit_r

# accumulate splits each cluster into units of at most nnz / UNITS nonzeros,
# and of at least MIN_UNIT_NNZ. The split depends on the data only, so the
# results do not depend on the thread count.
UNITS = 32
MIN_UNIT_NNZ = 1 << 16


def plan_units(indptr, cell_order, cluster_starts, target):
    """Return the units and buf_starts of K.accumulate.

    Units are runs of cell_order inside one cluster with about target
    nonzeros. Every cluster has at least one unit.
    """
    cum = np.concatenate(([0], np.cumsum(np.diff(indptr)[cell_order])))
    units = []
    buf_starts = [0]
    for k in range(len(cluster_starts) - 1):
        lo, hi = cluster_starts[k], cluster_starts[k + 1]
        nnz = cum[hi] - cum[lo]
        n = max(1, int(np.ceil(nnz / target)))
        cuts = np.searchsorted(cum, cum[lo] + nnz * np.arange(1, n) / n)
        edges = np.unique(np.concatenate(([lo], np.clip(cuts, lo, hi), [hi])))
        # An empty cluster (lo == hi) still needs a unit, which zeroes its row
        # of acc. run() never passes one, but fit_batch accepts it.
        if len(edges) == 1:
            edges = np.array([lo, hi])
        b0 = buf_starts[-1]
        for i in range(len(edges) - 1):
            units.append((k, edges[i], edges[i + 1], b0 + i - 1 if i else -1))
        buf_starts.append(b0 + len(edges) - 2)
    return (
        np.array(units, dtype=np.int64).reshape(-1, 4),
        np.array(buf_starts, dtype=np.int64),
    )


@dataclass
class BatchResult:
    """Output of fit_batch. counts has the sparsity pattern of the input."""

    contamination: np.ndarray
    counts: csr_matrix
    theta: np.ndarray
    delta: np.ndarray
    n_iter: int
    converged: bool
    final_change: float


class _State:
    """Arrays of one batch. The EM updates them in place."""

    def __init__(self, X, z, n_clusters, dtype):
        self.data = np.ascontiguousarray(X.data, dtype=dtype)
        # Native index dtype (int32 from scipy): half the bandwidth of int64.
        self.indices = np.ascontiguousarray(X.indices)
        self.indptr = np.ascontiguousarray(X.indptr)
        self.z = np.ascontiguousarray(z, dtype=np.int64)
        n_cells, n_genes = X.shape
        self.total = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()
        self.nonempty = self.total > 0
        self.cell_order = np.argsort(self.z, kind="stable").astype(np.int64)
        self.cluster_starts = np.zeros(n_clusters + 1, dtype=np.int64)
        np.cumsum(
            np.bincount(self.z, minlength=n_clusters), out=self.cluster_starts[1:]
        )
        self.units, self.buf_starts = plan_units(
            self.indptr,
            self.cell_order,
            self.cluster_starts,
            max(len(self.data) / UNITS, MIN_UNIT_NNZ),
        )
        self.bufs = np.empty((self.buf_starts[-1], n_genes))
        self.nc = np.empty(len(self.data), dtype=dtype)
        self.native = np.empty(n_cells)
        self.acc = np.empty((n_clusters, n_genes))
        self.phi = np.empty((n_clusters, n_genes))
        self.eta = np.empty((n_clusters, n_genes))

    def accumulate(self):
        K.accumulate(
            self.nc,
            self.indices,
            self.indptr,
            self.cell_order,
            self.units,
            self.buf_starts,
            self.bufs,
            self.acc,
        )

    def estep(self, theta):
        K.estep(
            self.data,
            self.indices,
            self.indptr,
            self.z,
            theta,
            self.phi,
            self.eta,
            self.nc,
            self.native,
        )


def fit_batch(
    X: csr_matrix,
    z: np.ndarray,
    n_clusters: int,
    *,
    delta: tuple[float, float],
    estimate_delta: bool,
    tol: float,
    max_iter: int,
    rng: np.random.Generator,
    dtype=np.float64,
    eta_fixed: np.ndarray | None = None,
    _theta0: np.ndarray | None = None,
) -> BatchResult:
    """Run the EM on X (cells x genes CSR) with 0-based cluster labels z.

    It follows celda's decontX: delta is refitted with MCMCprecision's
    defaults in each iteration, the EM stops when no native proportion changes
    by tol, and the reported contamination is that of the last iteration's
    E-step. The counts come from an E-step with the final parameters.

    _theta0 replaces the random initial theta. tests/test_celda.py,
    tests/test_kernels.py and tests/benchmarks/scripts/worker.py use it to start from
    given values, such as those captured from R, which is what makes R parity testable.
    """
    delta = np.asarray(delta, dtype=np.float64)
    if not X.data.any():
        # No cell has counts (negative values are rejected), so there is nothing
        # to fit: every cell gets NaN, as empty cells do.
        n_cells = X.shape[0]
        return BatchResult(
            contamination=np.full(n_cells, np.nan),
            counts=csr_matrix(
                (np.zeros(X.nnz, dtype=dtype), X.indices, X.indptr), shape=X.shape
            ),
            theta=np.full(n_cells, np.nan),
            delta=delta,
            n_iter=0,
            converged=True,
            final_change=0.0,
        )

    s = _State(X, z, n_clusters, dtype)
    estimate_eta = eta_fixed is None

    if _theta0 is not None:
        theta = np.array(_theta0, dtype=np.float64)
    else:
        theta = rng.beta(delta[0], delta[1], size=X.shape[0])

    # Initial profiles: theta-weighted counts with a pseudocount, as in celda.
    K.weighted(s.data, s.indptr, theta, s.nc)
    s.accumulate()
    K.profiles(s.acc + K.PSEUDOCOUNT, s.phi, s.eta, estimate_eta)
    if not estimate_eta:
        s.eta[:] = eta_fixed  # one profile, broadcast to every cluster

    total = s.total
    nonempty = s.nonempty
    all_nonempty = bool(nonempty.all())
    safe_total = np.where(nonempty, total, 1.0)
    change = np.inf
    converged = False
    n_iter = 0

    while n_iter < max_iter:
        n_iter += 1
        s.estep(theta)
        native_prop = s.native / safe_total
        contamination = 1.0 - native_prop
        if estimate_delta:
            delta = fit_r(native_prop if all_nonempty else native_prop[nonempty], delta)
        new = (s.native + delta[0]) / (total + delta[0] + delta[1])
        diff = new - theta if all_nonempty else new[nonempty] - theta[nonempty]
        change = float(np.max(np.abs(diff)))
        theta = new
        s.accumulate()
        K.profiles(s.acc, s.phi, s.eta, estimate_eta)
        if change < tol:
            converged = True
            break

    s.estep(theta)
    contamination[~nonempty] = np.nan

    # s is dropped on return, so its buffers can be handed over without copies.
    counts = csr_matrix((s.nc, X.indices, X.indptr), shape=X.shape)
    return BatchResult(
        contamination=contamination,
        counts=counts,
        theta=theta,
        delta=delta,
        n_iter=n_iter,
        converged=converged,
        final_change=change,
    )
