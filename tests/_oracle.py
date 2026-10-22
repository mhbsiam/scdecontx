"""A plain NumPy DecontX EM, as celda runs it. Tests compare the Numba kernels with it.

It shares no code with src/scdecontx except the Dirichlet fit, which
tests/test_dirichlet.py checks against R separately.
"""

import numpy as np

from scdecontx._dirichlet import fit_r

PC = 1e-20


def em(X, z, theta, *, tol=1e-3, max_iter=500, delta=(10, 10)):
    """Run the EM on a cells x genes CSR matrix with 0-based labels z.

    Return the contamination of the last iteration, the native counts at the
    nonzeros from the final parameters, theta, delta and the number of
    iterations.
    """
    X = X.tocsr()
    n_cells, n_genes = X.shape
    n_clusters = z.max() + 1
    rows = np.repeat(np.arange(n_cells), np.diff(X.indptr))
    cols = X.indices
    x = X.data.astype(float)
    k_nz = z[rows]
    total = np.bincount(rows, x, n_cells)
    delta = np.asarray(delta, float)

    def accum(values):
        acc = np.zeros((n_clusters, n_genes))
        np.add.at(acc, (k_nz, cols), values)
        return acc

    def normalize(acc):
        phi = acc / acc.sum(1, keepdims=True)
        other = acc.sum(0) - acc
        return phi, other / other.sum(1, keepdims=True)

    phi, eta = normalize(accum(x * theta[rows]) + PC)

    def estep(theta, phi, eta):
        pn = (phi[k_nz, cols] + PC) * (theta[rows] + PC)
        pc = (eta[k_nz, cols] + PC) * (1 - theta[rows] + PC)
        return x * (pn / (pn + pc))

    contamination_iter = None
    it = 0
    while it < max_iter:
        it += 1
        nat = estep(theta, phi, eta)
        native = np.bincount(rows, nat, n_cells)
        contamination_iter = 1 - native / total
        delta = fit_r(native / total, delta)
        new = (native + delta[0]) / (total + delta.sum())
        change = np.max(np.abs(new - theta))
        theta = new
        phi, eta = normalize(accum(nat))
        if change < tol:
            break
    return {
        "contamination_iter": contamination_iter,
        "native": estep(theta, phi, eta),
        "theta": theta,
        "delta": delta,
    }
