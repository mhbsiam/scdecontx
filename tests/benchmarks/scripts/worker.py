"""Run one DecontX implementation on one dataset. compare.py calls this in a subprocess.

impl is scdecontx, niruff or omicverse. A subprocess per implementation keeps
them apart.

Usage:
    python worker.py <impl> <src_path> <data_dir> <out.npz> <max_iter> <convergence>
        <seed> <repeats> [theta0.bin]

With theta0.bin, the port starts from that initial theta (R's own draw), not
from its own random draw. This removes the RNG difference between R and Python.
"""

import json
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix


def read_csr(data_dir, prefix):
    n_genes, n_cells = (
        int(v) for v in open(f"{data_dir}/{prefix}_dim.txt").read().split()
    )
    p = np.fromfile(f"{data_dir}/{prefix}_p.bin", dtype=np.int32)
    i = np.fromfile(f"{data_dir}/{prefix}_i.bin", dtype=np.int32)
    x = np.fromfile(f"{data_dir}/{prefix}_x.bin", dtype=np.float64)
    return csr_matrix((x, i, p), shape=(n_cells, n_genes))


def patch_theta0(impl, theta0):
    """Make the port's initial theta draw return theta0. Count the calls."""
    calls = {"n": 0}
    if impl == "niruff":
        import scipy.stats

        def rvs(a, b, size=None, random_state=None):
            calls["n"] += 1
            assert size == len(theta0)
            return theta0.copy()

        scipy.stats.beta.rvs = rvs
    else:

        class FakeRng:
            def beta(self, a, b, size=None):
                calls["n"] += 1
                assert size == len(theta0)
                return theta0.copy()

        np.random.default_rng = lambda seed=None: FakeRng()
    return calls


def main():
    impl, src, data_dir, out = sys.argv[1:5]
    max_iter, convergence, seed, repeats = (
        int(sys.argv[5]),
        float(sys.argv[6]),
        int(sys.argv[7]),
        int(sys.argv[8]),
    )
    theta0_path = sys.argv[9] if len(sys.argv) > 9 else None
    sys.path.insert(0, src)

    import anndata as ad
    import pandas as pd

    X = read_csr(data_dir, "counts")
    z = np.fromfile(f"{data_dir}/z.bin", dtype=np.float64).astype(int)
    adata = ad.AnnData(X=X)
    adata.obs["cl"] = pd.Categorical(z)

    theta0 = None if theta0_path is None else np.fromfile(theta0_path, dtype=np.float64)
    calls = None
    if theta0 is not None and not impl.startswith("scdecontx"):
        calls = patch_theta0(impl, theta0)

    kw = {"max_iter": max_iter, "convergence": convergence, "seed": seed}
    t_import = time.perf_counter()
    if impl == "scdecontx":
        import scdecontx as dx
        from scdecontx._em import fit_batch
        from scdecontx._validate import codes

        t_import = time.perf_counter() - t_import

        def run():
            if theta0 is None:
                res = dx.run(
                    adata,
                    "cl",
                    max_iter=max_iter,
                    tol=convergence,
                    random_state=seed,
                    copy=True,
                )
                fit = res.uns["scdecontx"]["fit"]
                info = {
                    "delta": [fit["delta_native"][0], fit["delta_contamination"][0]],
                    "n_iter": int(fit["n_iter"][0]),
                }
                return (
                    res.obs["scdecontx_contamination"].to_numpy(dtype=float),
                    csr_matrix(res.layers["scdecontx_counts"], dtype=np.float64),
                    info,
                )
            zz, n_clusters = codes(z)
            r = fit_batch(
                X,
                zz,
                n_clusters,
                delta=(10.0, 10.0),
                estimate_delta=True,
                tol=convergence,
                max_iter=max_iter,
                rng=np.random.default_rng(seed),
                _theta0=theta0,
            )
            info = {"delta": [float(d) for d in r.delta], "n_iter": int(r.n_iter)}
            return r.contamination, r.counts, info

    elif impl == "niruff":
        import decontx

        t_import = time.perf_counter() - t_import

        def run():
            res = decontx.decontx(
                adata, cluster_key="cl", copy=True, verbose=False, **kw
            )
            info = {}
            uns = res.uns["decontX"]
            if "fitted" in uns:
                info["delta"] = [float(v) for v in uns["fitted"]["delta"]]
                info["n_iter"] = int(uns["convergence"]["per_batch"]["n_iter"][0])
            else:
                # NiRuff stores the fitted delta under "parameters" and stores
                # no iteration count.
                info["delta"] = [float(v) for v in uns["parameters"]["delta"]]
            return (
                res.obs["decontX_contamination"].to_numpy(dtype=float),
                csr_matrix(res.layers["decontX_counts"], dtype=np.float64),
                info,
            )
    else:
        import pydecontx

        t_import = time.perf_counter() - t_import

        def run():
            res = pydecontx.decontx(adata, z=z, verbose=False, **kw)
            est = next(iter(res.estimates.values()))
            info = {
                "delta": [float(v) for v in est["delta"]],
                "n_iter": int(est["iteration"]),
            }
            return res.contamination, res.decontx_counts.T.tocsr(), info

    # The first call includes any JIT compilation.
    t0 = time.perf_counter()
    contamination, counts, info = run()
    cold = time.perf_counter() - t0

    warm = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        warm.append(time.perf_counter() - t0)

    counts.sort_indices()
    meta = {
        "import_s": t_import,
        "cold_s": cold,
        "warm_s": warm,
        "theta0_calls": None if calls is None else calls["n"],
        **info,
    }
    # Write to a temporary file, then rename, so no partial file stays at out.
    tmp = out + ".tmp.npz"
    np.savez(
        tmp,
        contamination=np.asarray(contamination, dtype=float),
        data=counts.data,
        indices=counts.indices,
        indptr=counts.indptr,
        shape=np.array(counts.shape),
        meta=json.dumps(meta),
    )
    os.replace(tmp, out)


if __name__ == "__main__":
    main()
