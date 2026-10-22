"""Real-data accuracy on the 10x human-mouse mixture (hgmm_12k).

Usage:
    python scripts/hgmm.py <hgmm_12k_raw_gene_bc_matrices_h5.h5>

Download (117 MB) into the git-ignored data/ folder:
    curl -L -o data/hgmm_12k/hgmm_12k_raw_gene_bc_matrices_h5.h5 \\
        https://cf.10xgenomics.com/samples/cell-exp/2.1.0/hgmm_12k/hgmm_12k_raw_gene_bc_matrices_h5.h5

Reads that align to only one species give the truth that DecontX can see.
In a human cell, every mouse count is contamination, and the reverse. Soup
of the cell's own species is invisible in the counts. Total contamination
is thus estimated as the cross-species fraction divided by the other
species' share of the soup. The empty droplets give that share. With
background=, DecontX gets the same droplets, so this total is a consistency
check against the soup, not an independent measurement.

Cells: barcodes with at least 10% of the 99th percentile of the expected
number of cells, as Cell Ranger 2 calls them. Multiplets: cells with
counts of both species above the 10th percentile of that species' counts
in the cells it dominates, as in Cell Ranger's GEM classification. The
script removes them, as in the DecontX paper. Empty droplets: barcodes
with 10 to 99 UMI.

scdecontx runs with the species as clusters (2 clusters, as in the paper)
and with k-means subclusters in each species (over-clustering), with and
without the empty droplets as background. Results: results/hgmm.{csv,md}.
"""

import sys
import time
import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy.cluster.vq import kmeans2
from scipy.sparse import csc_matrix, diags, hstack
from scipy.sparse.linalg import svds

import scdecontx as dx

RESULTS = Path(__file__).resolve().parent.parent / "results"
EXPECTED_CELLS = 12820  # Cell Ranger's call for this run
DROPLET_UMI = (10, 100)
SUBCLUSTERS = 4  # k-means clusters per species for the over-clustered labels
SEED = 0
METHODS = {
    "scdecontx": {"background": False},
    "scdecontx+bg": {"background": True},
}


def read(path):
    """Return barcodes x genes counts (human genes first) and the human gene mask."""
    with h5py.File(path, "r") as f:
        mats, names = [], []
        for genome in ("hg19", "mm10"):
            g = f[genome]
            m = csc_matrix(
                (g["data"][:], g["indices"][:], g["indptr"][:]),
                shape=tuple(g["shape"][:]),
            )
            mats.append(m.T.tocsr())
            names.append(g["genes"][:].astype(str))
        barcodes = f["hg19/barcodes"][:].astype(str)
        if not np.array_equal(barcodes, f["mm10/barcodes"][:].astype(str)):
            raise ValueError("The two genomes list different barcodes.")
    X = hstack(mats, format="csr").astype(np.float64)
    human = np.r_[np.ones(len(names[0]), bool), np.zeros(len(names[1]), bool)]
    return X, barcodes, np.concatenate(names), human


def classify(hg, mm):
    """Return the cells, their species (True = human), and the multiplets,
    as Cell Ranger 2 defines them."""
    total = hg + mm
    top = np.sort(total)[::-1][:EXPECTED_CELLS]
    cells = total >= 0.1 * np.percentile(top, 99)
    is_human = hg >= mm
    thr_hg = np.percentile(hg[cells & is_human], 10)
    thr_mm = np.percentile(mm[cells & ~is_human], 10)
    multiplet = cells & (hg >= thr_hg) & (mm >= thr_mm)
    return cells & ~multiplet, is_human, multiplet


def subclusters(X, species):
    """Cluster each species by k-means on 20 PCs of log-normalized counts."""
    labels = np.empty(X.shape[0], dtype=object)
    for name in np.unique(species):
        rows = np.flatnonzero(species == name)
        Y = X[rows]
        Y = diags(1e4 / np.asarray(Y.sum(axis=1)).ravel()) @ Y
        Y.data = np.log1p(Y.data)
        var = np.asarray(Y.power(2).mean(axis=0)).ravel() - np.square(
            np.asarray(Y.mean(axis=0)).ravel()
        )
        Y = Y[:, np.argsort(-var)[:2000]].toarray()
        Y -= Y.mean(axis=0)
        u, s, _ = svds(Y, k=20, random_state=SEED)
        _, k = kmeans2(u * s, SUBCLUSTERS, seed=SEED, minit="++")
        labels[rows] = [f"{name}_{i}" for i in k]
    return labels


def score(contam, counts, X, human_gene, is_human, soup_other):
    """Return metrics against the cross-species truth of each cell."""
    total = np.asarray(X.sum(axis=1)).ravel()
    other_gene = np.where(is_human[:, None], ~human_gene, human_gene)
    Xd, Cd = X.toarray(), counts.toarray()
    observed_other = np.where(other_gene, Xd, 0).sum(axis=1)
    kept_other = np.where(other_gene, Cd, 0).sum(axis=1)
    observed_own = np.where(other_gene, 0, Xd).sum(axis=1)
    removed_own = observed_own - np.where(other_gene, 0, Cd).sum(axis=1)
    foreign = observed_other / total
    # Derived from the same empty droplets that background mode receives, so
    # the total is conditional on the soup, not measured.
    total_est = foreign / soup_other
    out = {}
    for name, sel in (("all", slice(None)), ("human", is_human), ("mouse", ~is_human)):
        c, fg, te = contam[sel], foreign[sel], total_est[sel]
        out.update(
            {
                f"{name}_mean_contamination": float(c.mean()),
                f"{name}_mean_cross_species": float(fg.mean()),
                f"{name}_mean_total_est": float(te.mean()),
                f"{name}_pearson_cross": float(np.corrcoef(c, fg)[0, 1]),
                f"{name}_rmse_cross": float(np.sqrt(np.mean((c - fg) ** 2))),
                f"{name}_mae_total_est": float(np.mean(np.abs(c - te))),
                f"{name}_bias_total_est": float(np.mean(c - te)),
            }
        )
    out["species_leak"] = float(kept_other.sum() / observed_other.sum())
    out["own_removed"] = float(removed_own.sum() / observed_own.sum())
    # The own-species soup implied by the total estimate is the target of removal.
    out["own_soup_est"] = float(
        ((total_est - foreign) * total).clip(min=0).sum() / observed_own.sum()
    )
    return out


def main():
    path = sys.argv[1]
    X, barcodes, genes, human_gene = read(path)
    hg = np.asarray(X[:, human_gene].sum(axis=1)).ravel()
    mm = np.asarray(X[:, ~human_gene].sum(axis=1)).ravel()
    keep, is_human_all, multiplet = classify(hg, mm)
    total = hg + mm
    empty = (total >= DROPLET_UMI[0]) & (total < DROPLET_UMI[1])
    soup_mouse = mm[empty].sum() / total[empty].sum()
    print(
        f"{keep.sum()} cells ({(keep & is_human_all).sum()} human), "
        f"{multiplet.sum()} multiplets removed, {empty.sum()} empty droplets, "
        f"soup {soup_mouse:.3f} mouse",
        flush=True,
    )

    cells = np.flatnonzero(keep)
    Xc = X[cells]
    expressed = np.asarray(Xc.sum(axis=0)).ravel() > 0
    Xc = Xc[:, expressed]
    hgene = human_gene[expressed]
    is_human = is_human_all[cells]
    species = np.where(is_human, "human", "mouse")
    soup_other = np.where(is_human, soup_mouse, 1 - soup_mouse)

    adata = ad.AnnData(Xc)
    adata.obs_names = barcodes[cells]
    adata.var_names = genes[expressed]
    adata.obs["species"] = pd.Categorical(species)
    adata.obs["sub"] = pd.Categorical(subclusters(Xc, species))
    bg = ad.AnnData(X[np.flatnonzero(empty)][:, expressed])
    bg.obs_names = barcodes[empty]
    bg.var_names = adata.var_names

    rows = []
    for labels in ("species", "sub"):
        for method, kw in METHODS.items():
            a = adata.copy()
            start = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dx.run(
                    a,
                    labels,
                    background=bg if kw["background"] else None,
                    **{k: v for k, v in kw.items() if k != "background"},
                )
            seconds = time.perf_counter() - start
            m = score(
                a.obs["scdecontx_contamination"].to_numpy(),
                a.layers["scdecontx_counts"],
                Xc,
                hgene,
                is_human,
                soup_other,
            )
            m.update(
                clusters=labels,
                method=method,
                seconds=seconds,
                n_iter=int(a.uns["scdecontx"]["fit"]["n_iter"][0]),
            )
            rows.append(m)
            print(
                f"{labels:7s} {method:12s} r={m['all_pearson_cross']:.3f} "
                f"RMSE cross {m['all_rmse_cross']:.4f}  MAE total {m['all_mae_total_est']:.4f}"
                f"  leak {m['species_leak']:.3f}  own removed {m['own_removed']:.3f}"
                f" (soup {m['own_soup_est']:.3f})  {seconds:.1f} s",
                flush=True,
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "hgmm.csv", index=False, float_format="%.6g")
    (RESULTS / "hgmm.md").write_text(
        summary(df, keep.sum(), multiplet.sum(), empty.sum(), soup_mouse)
    )


def summary(df, n_cells, n_multiplets, n_empty, soup_mouse) -> str:
    cols = [
        ("all_pearson_cross", "r (cross-species)", "{:.3f}"),
        ("human_rmse_cross", "RMSE cross, human", "{:.4f}"),
        ("mouse_rmse_cross", "RMSE cross, mouse", "{:.4f}"),
        ("all_mae_total_est", "MAE total (est.)", "{:.4f}"),
        ("all_bias_total_est", "bias total (est.)", "{:+.4f}"),
        ("species_leak", "species leak", "{:.3f}"),
        ("own_removed", "own removed", "{:.3f}"),
    ]
    lines = [
        "# Human-mouse mixture (10x hgmm_12k)",
        "",
        f"Generated by `scripts/hgmm.py`. {n_cells} cells after removing "
        f"{n_multiplets} multiplets; {n_empty} empty droplets (10–99 UMI), whose "
        f"soup is {soup_mouse:.1%} mouse. Mean cross-species fraction "
        f"{df.all_mean_cross_species.iloc[0]:.4f}; mean estimated total "
        f"contamination {df.all_mean_total_est.iloc[0]:.4f}.",
        "",
        "*Cross* compares the contamination with each cell's fraction of "
        "other-species counts, the part DecontX can see. *Total (est.)* compares "
        "it with that fraction divided by the other species' share of the soup; "
        "background mode gets the same droplets, so this checks consistency with "
        "the soup, not an independently measured total. "
        "*Species leak* is the share of other-species counts kept in the "
        "corrected counts; *own removed* the share of own-species counts removed, "
        f"against {df.own_soup_est.iloc[0]:.3f} implied by the soup.",
        "",
        "| clusters | method | " + " | ".join(c[1] for c in cols) + " |",
        "|---" * (len(cols) + 2) + "|",
    ]
    for _, r in df.iterrows():
        vals = " | ".join(fmt.format(r[k]) for k, _, fmt in cols)
        lines.append(f"| {r.clusters} | {r.method} | {vals} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
