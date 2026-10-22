"""Compare scdecontx with R celda, NiRuff/decontx-python and omicverse/py-decontx.

Usage:
    python scripts/compare.py --niruff <path> --omicverse <path> --work <dir> [--quick]

<path> is a clone of each repository (scripts/setup_others.sh). R must have
celda installed. <work> holds the simulated data and every run, about 6 GB.
The script reuses a run that exists there only if its fingerprint (inputs,
parameters, repeats, source revision, scripts and package versions) matches;
otherwise it runs it again. --force reruns everything. The results go to
results/results.json. The per-cell estimates of the datasets in CELL_EXPORT
go to results/cells/. scripts/report.py makes the tables and figures from
them.

Dataset groups:
- accuracy: celda::simulateContamination, 2,000 cells x 1,000 genes, known truth.
- real: PBMC3k (10x), clustered with Leiden. No truth.
- speed: 5,000 to 20,000 cells x 2,000 genes, and 20,000 cells x 15,000 genes.
- sweep_g1000 / sweep_g15000: contamination from ~9% to ~50%, three seeds each,
  at 1,000 genes and at a realistic 15,000 genes.
- merged: 16 simulated cell types given to the methods as 8 clusters. Each
  cluster mixes two types, as coarse clusters of real data do.

Every implementation runs at celda's defaults: convergence=0.001, max_iter 500.
"shared" runs start from R's initial theta, captured with trace(). This removes
the RNG difference.
"""

import argparse
import functools
import hashlib
import inspect
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from worker import read_csr  # noqa: E402

# (max_iter, convergence)
R_RULE = (500, 0.001)
SEED = 12345
RULE = {
    "scdecontx": R_RULE,
    "niruff": R_RULE,
    "omicverse": R_RULE,
}

# name, C, G, K, sim seed, delta. delta = (1, 10) gives ~9% contamination; (5, 5) ~50%.
LEVELS = [(1, 10), (2, 8), (3, 7), (4, 6), (5, 5)]
GROUPS = {
    "accuracy": [
        ("sim_low_s1", 2000, 1000, 5, 1, (1, 10)),
        ("sim_low_s2", 2000, 1000, 5, 2, (1, 10)),
        ("sim_low_s3", 2000, 1000, 5, 3, (1, 10)),
        ("sim_high_s1", 2000, 1000, 5, 1, (4, 6)),
        ("sim_high_s2", 2000, 1000, 5, 2, (4, 6)),
    ],
    "real": [("pbmc3k",)],
    "speed": [
        (f"speed_{c // 1000}k", c, 2000, 8, 7, (1, 10)) for c in (5000, 10000, 20000)
    ]
    + [("speed_20k_g15k", 20000, 15000, 8, 7, (1, 10))],
    "sweep_g1000": [
        (f"sweep_d{a}_{b}_s{s}", 2000, 1000, 5, s, (a, b))
        for a, b in LEVELS
        for s in (11, 12, 13)
    ],
    "sweep_g15000": [
        (f"sweep15k_d{a}_{b}_s{s}", 3000, 15000, 8, s, (a, b))
        for a, b in LEVELS
        for s in (21, 22, 23)
    ],
    "merged": [
        (f"merged_d{a}_{b}_s{s}", 3000, 15000, 16, s, (a, b))
        for a, b in [(1, 10), (4, 6)]
        for s in (41, 42)
    ],
}
# Warm calls timed per dataset; 0 for groups whose timings are not reported.
REPEATS = {"speed": 3, "accuracy": 2, "real": 3}
# Set by --force: ignore every cached run.
FORCE = False
# The per-cell estimates of these datasets feed the figures.
CELL_EXPORT = ("sweep15k_d4_6_s21", "pbmc3k")
QUICK = {
    "accuracy": 1,
    "real": 1,
    "speed": 1,
    "sweep_g1000": 1,
    "sweep_g15000": 1,
    "merged": 1,
}


def runs_for(group):
    """Return the (impl, shared, threads) Python runs of a group."""
    impls = ["scdecontx", "niruff", "omicverse"]
    if group in ("accuracy", "real"):
        return [(i, s, None) for i in impls for s in (False, True)]
    if group == "speed":
        return [(i, False, None) for i in impls] + [("scdecontx", False, 1)]
    return [(i, False, None) for i in impls]


def sh(cmd, env=None):
    res = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    if res.returncode != 0:
        print(res.stdout, res.stderr)
        raise RuntimeError(f"failed: {' '.join(map(str, cmd))}")
    return res.stdout


# Cache fingerprints ---------------------------------------------------------


def file_hash(*paths):
    """Hash of the names and contents of files, in sorted order."""
    h = hashlib.sha256()
    for path in sorted(Path(p) for p in paths):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


@functools.cache
def source_state(path):
    """Revision of an implementation: its git commit, if any, and a hash of
    its Python files, which also covers uncommitted changes."""
    res = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    files = [f for f in Path(path).rglob("*.py") if ".git" not in f.parts]
    return {
        "commit": res.stdout.strip() if res.returncode == 0 else None,
        "py_files": file_hash(*files),
    }


@functools.cache
def py_versions():
    out = {
        "python": platform.python_version(),
        "machine": f"{platform.platform()} {platform.machine()}, {os.cpu_count()} cpus",
    }
    for pkg in ("numpy", "scipy", "numba", "anndata", "pandas", "scanpy", "igraph"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


@functools.cache
def r_versions():
    out = sh(
        [
            "Rscript",
            "-e",
            'cat(R.version.string, as.character(packageVersion("celda")), sep="\\n")',
        ]
    ).splitlines()
    return {"R": out[0], "celda": out[1]}


def fingerprint(**kw):
    """The fingerprint as it reads back from JSON (tuples become lists)."""
    return json.loads(json.dumps(kw, sort_keys=True))


def cached(path, fp):
    """True if the run whose fingerprint is at path was made with fp."""
    if FORCE or not path.exists():
        return False
    return json.loads(path.read_text()) == fp


def save_fingerprint(path, fp):
    path.write_text(json.dumps(fp, indent=1, sort_keys=True))


def data_id(data):
    """Identity of a prepared dataset, for the fingerprints of its runs."""
    return file_hash(data / "fingerprint.json")


def check_repeats(times, repeats, what):
    """Raise if a run timed a different number of calls than requested."""
    n = np.atleast_1d(times).size
    if repeats and n != repeats:
        raise RuntimeError(f"{what} has {n} timed calls, not {repeats}.")


# Data and runs ---------------------------------------------------------------


def _simulate_to(out, fp, C, G, K, seed, delta):
    if cached(out / "fingerprint.json", fp):
        return
    shutil.rmtree(out, ignore_errors=True)
    sh(["Rscript", HERE / "simulate.R", out, C, G, K, seed, *delta])
    save_fingerprint(out / "fingerprint.json", fp)


def simulate(work, name, C, G, K, seed, delta):
    out = work / "data" / name
    fp = fingerprint(
        kind="simulate",
        args=[C, G, K, seed, list(delta)],
        scripts=file_hash(HERE / "simulate.R", HERE / "io.R"),
        r=r_versions(),
    )
    if not name.startswith("merged"):
        if not cached(out / "fingerprint.json", fp):
            print(f"  simulate {name}", flush=True)
            _simulate_to(out, fp, C, G, K, seed, delta)
        return out
    # K cell types; the methods get K / 2 clusters.
    merged_fp = fingerprint(**fp, merged=True)
    if cached(out / "fingerprint.json", merged_fp):
        return out
    print(f"  simulate {name}", flush=True)
    src = work / "data" / (name + "_types")
    _simulate_to(src, fp, C, G, K, seed, delta)
    shutil.rmtree(out, ignore_errors=True)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(src, tmp)
    z = np.fromfile(src / "z.bin")
    ((z - 1) // 2 + 1).tofile(tmp / "z.bin")
    save_fingerprint(tmp / "fingerprint.json", merged_fp)
    tmp.rename(out)
    return out


def prepare_pbmc(work, h5ad):
    """Filter genes and cluster PBMC3k with Leiden. Save the raw counts."""
    out = work / "data" / "pbmc3k"
    fp = fingerprint(
        kind="pbmc3k",
        h5ad=file_hash(h5ad),
        code=hashlib.sha256(inspect.getsource(prepare_pbmc).encode()).hexdigest()[:16],
        python=py_versions(),
    )
    if cached(out / "fingerprint.json", fp):
        return out
    shutil.rmtree(out, ignore_errors=True)
    import anndata as ad
    import scanpy as sc

    print("  prepare pbmc3k", flush=True)
    adata = ad.read_h5ad(h5ad)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    raw = csr_matrix(adata.X, dtype=np.float64)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    sc.pp.pca(adata, n_comps=30, mask_var="highly_variable", random_state=0)
    sc.pp.neighbors(adata, random_state=0)
    sc.tl.leiden(adata, resolution=0.5, flavor="igraph", n_iterations=2, random_state=0)
    z = adata.obs["leiden"].cat.codes.to_numpy() + 1

    out.mkdir(parents=True, exist_ok=True)
    raw.sort_indices()
    (out / "counts_dim.txt").write_text(f"{raw.shape[1]}\n{raw.shape[0]}\n")
    raw.indptr.astype(np.int32).tofile(out / "counts_p.bin")
    raw.indices.astype(np.int32).tofile(out / "counts_i.bin")
    raw.data.astype(np.float64).tofile(out / "counts_x.bin")
    z.astype(np.float64).tofile(out / "z.bin")
    save_fingerprint(out / "fingerprint.json", fp)
    return out


def run_r(work, name, data, repeats):
    out = work / "r" / name
    n = max(repeats, 1)
    fp = fingerprint(
        data=data_id(data),
        rule=list(R_RULE),
        seed=SEED,
        repeats=n,
        scripts=file_hash(HERE / "run_r.R", HERE / "io.R"),
        r=r_versions(),
    )
    if not (cached(out / "fingerprint.json", fp) and (out / "meta.json").exists()):
        print(f"  R {name}", flush=True)
        shutil.rmtree(out, ignore_errors=True)
        sh(["Rscript", HERE / "run_r.R", data, out, *R_RULE, SEED, n])
        save_fingerprint(out / "fingerprint.json", fp)
    meta = json.loads((out / "meta.json").read_text())
    check_repeats(meta["em_s"], n, f"R run of {name}")
    return out


def run_py(work, name, impl, src, data, repeats, theta0=None, threads=None):
    tag = impl + ("_1t" if threads == 1 else "") + ("_shared" if theta0 else "")
    out = work / "py" / name / f"{tag}.npz"
    fp_path = out.with_suffix(".fingerprint.json")
    fp = fingerprint(
        impl=impl,
        data=data_id(data),
        rule=list(RULE[impl]),
        seed=SEED,
        repeats=repeats,
        threads=threads,
        theta0=file_hash(theta0) if theta0 else None,
        source=source_state(src),
        worker=file_hash(HERE / "worker.py"),
        python=py_versions(),
    )
    if not (cached(fp_path, fp) and out.exists()):
        print(f"  {tag} {name}", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        env = os.environ.copy()
        if threads:
            env["NUMBA_NUM_THREADS"] = str(threads)
        cmd = [sys.executable, HERE / "worker.py", impl, src, data, out]
        cmd += [*RULE[impl], SEED, repeats]
        if theta0:
            cmd.append(theta0)
        sh(cmd, env=env)
        save_fingerprint(fp_path, fp)
    d = np.load(out)
    counts = csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    meta = json.loads(str(d["meta"]))
    check_repeats(meta["warm_s"], repeats, f"{tag} run of {name}")
    meta["fingerprint"] = file_hash(fp_path)
    return tag, d["contamination"], counts, meta


def pearson(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def rel_fro(a, b):
    return float(np.sqrt((a - b).power(2).sum() / b.power(2).sum()))


def metrics(
    contam, counts, X, ref_contam=None, ref_counts=None, true_contam=None, native=None
):
    tot = np.asarray(X.sum(1)).ravel()
    m = {
        # Contamination must equal 1 - (decontaminated total / observed total).
        "self_consistency": float(
            np.nanmax(np.abs(contam - (1 - np.asarray(counts.sum(1)).ravel() / tot)))
        ),
        "mean_contamination": float(np.nanmean(contam)),
        "frac_nonzero_kept": float(counts.count_nonzero() / X.count_nonzero()),
    }
    if ref_contam is not None:
        m["r_contam_pearson"] = pearson(contam, ref_contam)
        m["r_contam_maxdiff"] = float(np.nanmax(np.abs(contam - ref_contam)))
    if ref_counts is not None:
        m["r_counts_maxdiff"] = float(abs(counts - ref_counts).max())
    if true_contam is not None:
        m["true_contam_pearson"] = pearson(contam, true_contam)
        m["true_contam_mae"] = float(np.nanmean(np.abs(contam - true_contam)))
        m["true_contam_bias"] = float(np.nanmean(contam - true_contam))
        m["true_native_relfro"] = rel_fro(counts, native)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--niruff", required=True)
    ap.add_argument("--omicverse", required=True)
    ap.add_argument("--work", required=True)
    # Default: results/. With --quick, <work>/results, which does not replace it.
    ap.add_argument("--out")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--force", action="store_true", help="rerun every run, ignoring the cache"
    )
    ap.add_argument(
        "--only",
        nargs="+",
        metavar="DATASET",
        help="run only these datasets and merge them into an existing results.json",
    )
    args = ap.parse_args()
    global FORCE
    FORCE = args.force
    work = Path(args.work)
    if args.out:
        out = Path(args.out)
    else:
        out = work / "results" if args.quick else HERE.parent / "results"
    (out / "cells").mkdir(parents=True, exist_ok=True)
    src = {
        "scdecontx": str(HERE.parents[2] / "src"),
        "niruff": args.niruff,
        "omicverse": args.omicverse,
    }
    results = {
        "settings": {
            "r_rule": {"max_iter": R_RULE[0], "convergence": R_RULE[1]},
            "seed": SEED,
            "repeats": REPEATS,
            "python": py_versions(),
            "r": r_versions(),
            "sources": {impl: source_state(path) for impl, path in src.items()},
        },
        "datasets": {},
    }
    if args.only and (out / "results.json").exists():
        results["datasets"] = json.loads((out / "results.json").read_text())["datasets"]

    for group, specs in GROUPS.items():
        if args.quick:
            specs = specs[: QUICK[group]]
        py_runs = runs_for(group)
        repeats = REPEATS.get(group, 0)
        for spec in specs:
            name = spec[0]
            if args.only and name not in args.only:
                continue
            print(name, flush=True)
            if group == "real":
                data = prepare_pbmc(
                    work, Path(args.omicverse) / "data" / "pbmc3k_raw.h5ad"
                )
            else:
                data = simulate(work, *spec)
            X = read_csr(data, "counts")
            true_contam = native = None
            if group != "real":
                true_contam = np.fromfile(data / "true_contamination.bin")
                native = read_csr(data, "native")

            entry = {
                "group": group,
                "n_cells": X.shape[0],
                "n_genes": X.shape[1],
                "nnz": int(X.nnz),
                "n_clusters": int(len(np.unique(np.fromfile(data / "z.bin")))),
            }
            if group != "real":
                entry["true_mean"] = float(true_contam.mean())
                entry["delta"] = list(spec[5])
                entry["observed_vs_native_relfro"] = rel_fro(X, native)

            rdir = run_r(work, name, data, repeats)
            r_contam = np.fromfile(rdir / "contamination.bin")
            r_counts = read_csr(rdir, "decontx")
            entry["R"] = {
                **json.loads((rdir / "meta.json").read_text()),
                "fingerprint": file_hash(rdir / "fingerprint.json"),
                **metrics(
                    r_contam, r_counts, X, true_contam=true_contam, native=native
                ),
            }

            cells = {"R": r_contam}
            for impl, shared, threads in py_runs:
                theta0 = str(rdir / "theta0.bin") if shared else None
                tag, c, m, meta = run_py(
                    work, name, impl, src[impl], data, repeats, theta0, threads
                )
                entry[tag] = {
                    **meta,
                    **metrics(c, m, X, r_contam, r_counts, true_contam, native),
                }
                if not shared and threads is None:
                    cells[tag] = c
            results["datasets"][name] = entry
            (out / "results.json").write_text(json.dumps(results, indent=1))
            if name in CELL_EXPORT:
                export_cells(
                    out / "cells" / f"{name}.csv.gz", data, X, true_contam, cells
                )

    print(f"wrote {out / 'results.json'}")


def export_cells(path, data, X, true_contam, cells):
    """Write the cluster, total counts, truth and each tool's estimate per cell."""
    import pandas as pd

    table = pd.DataFrame(
        {
            "cluster": np.fromfile(data / "z.bin").astype(int),
            "total_counts": np.asarray(X.sum(axis=1)).ravel().astype(int),
        }
    )
    if true_contam is not None:
        table["true_contamination"] = true_contam
    for tool, values in cells.items():
        table[tool] = values
    table.to_csv(path, index_label="cell", float_format="%.6g")


if __name__ == "__main__":
    main()
