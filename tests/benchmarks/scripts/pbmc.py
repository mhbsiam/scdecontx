"""The PBMC test of the DecontX paper: pbmc4k against flow-sorted PBMCs.

Usage:
    python scripts/pbmc.py <data/pbmc4k/pbmc4k_raw_gene_bc_matrices_h5.h5> <data/sorted_pbmc>

Download into the git-ignored data/ folder:
    curl -L -o data/pbmc4k/pbmc4k_raw_gene_bc_matrices_h5.h5 \\
        https://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc4k/pbmc4k_raw_gene_bc_matrices_h5.h5
    for d in b_cells cd14_monocytes cd56_nk cd4_t_helper regulatory_t naive_t \\
             naive_cytotoxic cytotoxic_t cd34; do
        curl -L -o data/sorted_pbmc/$d.tar.gz \\
            https://cf.10xgenomics.com/samples/cell-exp/1.1.0/$d/${d}_filtered_gene_bc_matrices.tar.gz
    done

Each sorted population ran in its own 10x channel (Zheng et al. 2017). Its
cells contain little RNA of the other types. The share of its cells with
counts of another type's markers is the reference. pbmc4k ran in one
channel. This script clusters its cells with Leiden and names each cluster
by its markers. It then runs DecontX on the Leiden clusters and on the
broad types, with and without the empty droplets (10-99 UMI) as background.
A good correction brings the share of cells with other types' markers down
to the sorted reference and keeps the share with their own markers.
Results: results/pbmc.{csv,md}.
"""

import sys
import tarfile
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csc_matrix

import scdecontx as dx

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED = 0
EXPECTED_CELLS = 4340  # Cell Ranger's call for pbmc4k, as in the paper
DROPLET_UMI = (10, 100)
MARKERS = {
    "T": ["CD3D", "CD3E"],
    "B": ["CD79A", "CD79B", "MS4A1"],
    "monocyte": ["LYZ", "S100A8", "S100A9"],
    "NK": ["GNLY", "NKG7"],
}
SORTED = {
    "b_cells": "B",
    "cd14_monocytes": "monocyte",
    "cd56_nk": "NK",
    "cd4_t_helper": "T",
    "regulatory_t": "T",
    "naive_t": "T",
    "naive_cytotoxic": "T",
    "cytotoxic_t": "T",
}
METHODS = {
    "scdecontx": {},
    "scdecontx+bg": {"background": True},
}


def read_10x_h5(path):
    with h5py.File(path, "r") as f:
        g = f["GRCh38"]
        m = csc_matrix(
            (g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:])
        )
        barcodes = g["barcodes"][:].astype(str)
        genes = g["gene_names"][:].astype(str)
    return m.T.tocsr().astype(np.float64), barcodes, genes


def read_sorted(folder):
    """Return (cell type, cells x genes counts, genes) per sorted population."""
    out = []
    for name, cell_type in SORTED.items():
        with tarfile.open(Path(folder) / f"{name}.tar.gz") as tar:
            base = "filtered_matrices_mex/hg19/"
            m = mmread(tar.extractfile(base + "matrix.mtx")).T.tocsr()
            genes = pd.read_csv(
                tar.extractfile(base + "genes.tsv"), sep="\t", header=None
            )[1].to_numpy()
        out.append((cell_type, m, genes))
    return out


def marker_shares(X, genes, cell_type, threshold=1.0):
    """For each type and marker set: share of cells with at least threshold
    marker counts, and mean marker counts per 10,000 counts of the cell
    (key "cp10k")."""
    rows = {}
    index = {g: i for i, g in enumerate(genes)}
    size = np.asarray(X.sum(axis=1)).ravel()
    for own in MARKERS:
        sel = np.flatnonzero(cell_type == own)
        if not len(sel):
            continue
        for other, markers in MARKERS.items():
            cols = [index[m] for m in markers if m in index]
            counts = np.asarray(X[sel][:, cols].sum(axis=1)).ravel()
            rows[(own, other)] = float(np.mean(counts >= threshold))
            rows[(own, other, "cp10k")] = float(np.mean(1e4 * counts / size[sel]))
    return rows


def cluster(X, genes):
    a = ad.AnnData(X.copy())
    a.var_names = pd.Index(genes).astype(str)
    a.var_names_make_unique()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000)
    sc.pp.pca(a, n_comps=30, mask_var="highly_variable", random_state=SEED)
    sc.pp.neighbors(a, random_state=SEED)
    sc.tl.leiden(a, resolution=0.5, random_state=SEED, flavor="igraph", n_iterations=2)
    leiden = a.obs["leiden"].to_numpy()
    # Name each cluster by its strongest marker set (mean log expression).
    score = pd.DataFrame(
        {
            t: np.asarray(
                a[:, [m for m in ms if m in a.var_names]].X.mean(axis=1)
            ).ravel()
            for t, ms in MARKERS.items()
        }
    )
    score["leiden"] = leiden
    best = score.groupby("leiden").mean().idxmax(axis=1)
    return leiden, best.loc[leiden].to_numpy()


def main():
    X, barcodes, genes = read_10x_h5(sys.argv[1])
    total = np.asarray(X.sum(axis=1)).ravel()
    top = np.sort(total)[::-1][:EXPECTED_CELLS]
    is_cell = total >= 0.1 * np.percentile(top, 99)
    empty = (total >= DROPLET_UMI[0]) & (total < DROPLET_UMI[1])
    cells = X[np.flatnonzero(is_cell)]
    expressed = np.asarray(cells.sum(axis=0)).ravel() > 0
    cells = cells[:, expressed]
    genes = genes[expressed]
    leiden, cell_type = cluster(cells, genes)
    print(
        f"{cells.shape[0]} cells, {len(np.unique(leiden))} Leiden clusters, "
        f"types {pd.Series(cell_type).value_counts().to_dict()}, {empty.sum()} droplets",
        flush=True,
    )

    ref = {}
    for t, m, g in read_sorted(sys.argv[2]):
        for key, v in marker_shares(m, g, np.full(m.shape[0], t)).items():
            ref.setdefault(key, []).append((v, m.shape[0]))
    ref = {
        k: np.average([v for v, _ in x], weights=[n for _, n in x])
        for k, x in ref.items()
    }

    adata = ad.AnnData(cells)
    adata.obs_names = barcodes[is_cell]
    adata.var_names = pd.Index(genes).astype(str)
    adata.var_names_make_unique()
    adata.obs["leiden"] = pd.Categorical(leiden)
    adata.obs["type"] = pd.Categorical(cell_type)
    bg = ad.AnnData(X[np.flatnonzero(empty)][:, expressed])
    bg.obs_names = barcodes[empty]
    bg.var_names = adata.var_names

    rows = [{"labels": "-", "method": "raw", **marker_shares(cells, genes, cell_type)}]
    for labels in ("type", "leiden"):
        for method, kw in METHODS.items():
            a = adata.copy()
            kw = dict(kw)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dx.run(
                    a,
                    labels,
                    random_state=SEED,
                    background=bg if kw.pop("background", False) else None,
                    **kw,
                )
            shares = marker_shares(a.layers["scdecontx_counts"], genes, cell_type)
            rows.append(
                {
                    "labels": labels,
                    "method": method,
                    "median_contamination": float(
                        np.median(a.obs["scdecontx_contamination"])
                    ),
                    **shares,
                }
            )
            print(labels, method, flush=True)
    df = pd.DataFrame(rows)
    pairs = [k for k in ref if len(k) == 2 and k[0] != k[1]]
    own = [k for k in ref if len(k) == 2 and k[0] == k[1]]
    df["foreign_error"] = [
        np.mean([abs(r[k] - ref[k]) for k in pairs]) for _, r in df.iterrows()
    ]
    df["own_kept"] = [np.mean([r[k] for k in own]) for _, r in df.iterrows()]

    # Marker counts per 10k: mean |log10| ratio to the sorted reference, with
    # 0.1 per 10k added to both. Depth normalization makes the older chemistry
    # of the sorted runs matter less than for the shares above.
    def log_error(r, keys):
        return float(
            np.mean(
                [
                    abs(
                        np.log10(
                            (r[k + ("cp10k",)] + 0.1) / (ref[k + ("cp10k",)] + 0.1)
                        )
                    )
                    for k in keys
                ]
            )
        )

    df["foreign_cp10k_error"] = [log_error(r, pairs) for _, r in df.iterrows()]
    df["own_cp10k_error"] = [log_error(r, own) for _, r in df.iterrows()]
    out = df.copy()
    out.columns = [
        c
        if isinstance(c, str)
        else f"{c[1]} markers in {c[0]}" + (" cp10k" if len(c) == 3 else "")
        for c in out.columns
    ]
    out.to_csv(RESULTS / "pbmc.csv", index=False, float_format="%.6g")
    (RESULTS / "pbmc.md").write_text(summary(df, ref, pairs, own))


def summary(df, ref, pairs, own) -> str:
    show = [p for p in pairs if ref[p] < 0.5]
    head = [
        "labels",
        "method",
        "median contamination",
        "foreign error",
        "own kept",
        "foreign cp10k error",
        "own cp10k error",
    ] + [f"{o} in {t}" for t, o in show]
    lines = [
        "# pbmc4k against sorted PBMCs",
        "",
        "Generated by `scripts/pbmc.py`. Each column is the share of the cells of "
        "one type with at least one count of another type's markers (T: CD3D, "
        "CD3E; B: CD79A, CD79B, MS4A1; monocyte: LYZ, S100A8, S100A9; NK: GNLY, "
        "NKG7). *Sorted* is the same share in the flow-sorted populations, each "
        "run in its own channel. *Foreign error* is the mean absolute difference "
        "from the sorted shares over all pairs of types; *own kept* the mean "
        "share of cells with their own type's markers (sorted: "
        f"{np.mean([ref[k] for k in own]):.3f}). The *cp10k errors* compare marker "
        "counts per 10,000 counts of the cell with the sorted cells: the mean "
        "absolute log10 ratio (0.1 per 10k added), for other types' markers and "
        "for the cell's own.",
        "",
        "| " + " | ".join(head) + " |",
        "|---" * len(head) + "|",
        "| sorted | reference | | | | | | "
        + " | ".join(f"{ref[p]:.3f}" for p in show)
        + " |",
    ]
    for _, r in df.iterrows():
        mc = r.get("median_contamination")
        mc = "" if pd.isna(mc) else f"{mc:.3f}"
        vals = [
            r["labels"],
            r["method"],
            mc,
            f"{r['foreign_error']:.3f}",
            f"{r['own_kept']:.3f}",
            f"{r['foreign_cp10k_error']:.3f}",
            f"{r['own_cp10k_error']:.3f}",
        ]
        vals += [f"{r[p]:.3f}" for p in show]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
