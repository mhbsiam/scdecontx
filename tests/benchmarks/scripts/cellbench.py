"""The CellBench datasets of the DecontX paper (Tian et al. 2019, sc_mixology).

Usage:
    python scripts/cellbench.py <data/cellbench>

Download into the git-ignored data/ folder and export with R first:
    for f in mRNAmix_qc.RData sincell_with_class.RData sincell_with_class_5cl.RData; do
        curl -L -o data/cellbench/$f \\
            https://raw.githubusercontent.com/LuyiTian/sc_mixology/master/data/$f
    done
    Rscript tests/benchmarks/scripts/export_cellbench.R data/cellbench

RNA mixtures (CEL-seq2 and SORT-seq): RNA of three lung adenocarcinoma cell
lines, mixed in known proportions and diluted to single-cell amounts. As in
the paper, the cluster of a pseudo-cell is the line with more than half of
its RNA. The equal 1/3 mixtures have no such line and are left out. The
expected contamination is the share of the other lines: 0 or 0.32. Library
preparation adds its own contamination, so the estimate can exceed the
design.

Single cells (10x, CEL-seq2, Drop-seq; 3 or 5 lines): demuxlet labels the
cell line and the doublets. As in the paper, the script fits all cells
with the demuxlet lines as clusters; cells without a line are left out.
There is no truth per cell. The
script compares the median contamination with the paper. The doublets
hold RNA of two lines, so they should rank above the singlets (AUROC).

The script removes ERCC spike-ins. No empty droplets are available, so
there is no background mode. Results: results/cellbench.{csv,md}.
"""

import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parent))
from worker import read_csr

import scdecontx as dx

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEED = 12345
METHODS = {
    "scdecontx": {},
}
MIXTURES = {"sce2_qc": "CEL-seq2", "sce8_qc": "SORT-seq"}
# name: (protocol, lines, the paper's median contamination)
SINGLE = {
    "sce_sc_10x_qc": ("10x", 3, 0.0881),
    "sce_sc_CELseq2_qc": ("CEL-seq2", 3, 0.1099),
    "sce_sc_Dropseq_qc": ("Drop-seq", 3, 0.0929),
    "sce_sc_10x_5cl_qc": ("10x", 5, 0.0496),
    "sc_Celseq2_5cl_p1": ("CEL-seq2 p1", 5, 0.1313),
    "sc_Celseq2_5cl_p2": ("CEL-seq2 p2", 5, 0.1396),
    "sc_Celseq2_5cl_p3": ("CEL-seq2 p3", 5, 0.0922),
}
# The paper's medians for the RNA mixtures: (68/16/16, pure).
PAPER_MIX = {"CEL-seq2": (0.4115, 0.1808), "SORT-seq": (0.3647, 0.0909)}


def load(export, name):
    d = export / name
    X = read_csr(d, "counts")
    genes = np.array((d / "genes.txt").read_text().split())
    cells = pd.read_csv(d / "cells.csv")
    keep = ~pd.Series(genes).str.startswith("ERCC").to_numpy()
    return X[:, keep].tocsr(), cells


def fit(X, labels, **kw):
    a = ad.AnnData(X)
    # No cast to str: it would turn missing labels into a "nan" cluster, which
    # dx.run rejects otherwise.
    a.obs["cluster"] = pd.Categorical(np.asarray(labels))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dx.run(a, "cluster", random_state=SEED, **kw)
    return a.obs["scdecontx_contamination"].to_numpy()


def mixtures(export):
    rows = []
    for name, protocol in MIXTURES.items():
        X, cells = load(export, name)
        props = cells[["H2228_prop", "H1975_prop", "HCC827_prop"]].to_numpy()
        main = props.max(axis=1)
        keep = main > 0.5
        X, props, main = X[np.flatnonzero(keep)], props[keep], main[keep]
        expected = 1 - main
        labels = props.argmax(axis=1)
        for method, kw in METHODS.items():
            c = fit(X, labels, **kw)
            pure, mixed = expected == 0, expected > 0
            rows.append(
                {
                    "dataset": f"RNA mix, {protocol}",
                    "method": method,
                    "n": int(keep.sum()),
                    "pearson_design": float(np.corrcoef(c, expected)[0, 1]),
                    "median_pure": float(np.median(c[pure])),
                    "median_68": float(np.median(c[mixed])),
                    # contamination added to the design: the median of the
                    # mixtures minus the median of the pure cells
                    "gap": float(np.median(c[mixed]) - np.median(c[pure])),
                    "paper_median_68": PAPER_MIX[protocol][0],
                    "paper_median_pure": PAPER_MIX[protocol][1],
                }
            )
            print(rows[-1], flush=True)
    return rows


def single_cells(export):
    rows = []
    for name, (protocol, n_lines, paper) in SINGLE.items():
        X, cells = load(export, name)
        labelled = cells["cell_line_demuxlet"].notna().to_numpy()
        X, cells = X[np.flatnonzero(labelled)], cells[labelled]
        doublet = (cells["demuxlet_cls"] == "DBL").to_numpy()
        for method, kw in METHODS.items():
            c = fit(X, cells["cell_line_demuxlet"], **kw)
            auc = np.nan
            if doublet.any():
                u = mannwhitneyu(c[doublet], c[~doublet]).statistic
                auc = float(u / (doublet.sum() * (~doublet).sum()))
            rows.append(
                {
                    "dataset": f"{n_lines} lines, {protocol}",
                    "method": method,
                    "n": int(len(c)),
                    "n_doublets": int(doublet.sum()),
                    "median_singlets": float(np.median(c[~doublet])),
                    "paper_median": paper,
                    "doublet_auroc": auc,
                }
            )
            print(rows[-1], flush=True)
    return rows


def table(df, cols):
    head = "| dataset | method | " + " | ".join(label for _, label, _ in cols) + " |"
    lines = [head, "|---" * (len(cols) + 2) + "|"]
    for _, r in df.iterrows():
        vals = " | ".join(fmt.format(r[k]) for k, _, fmt in cols)
        lines.append(f"| {r.dataset} | {r.method} | {vals} |")
    return lines


def main():
    export = Path(sys.argv[1]) / "export"
    mix = pd.DataFrame(mixtures(export))
    sc = pd.DataFrame(single_cells(export))
    pd.concat([mix.assign(kind="mixture"), sc.assign(kind="single")]).to_csv(
        RESULTS / "cellbench.csv", index=False, float_format="%.6g"
    )
    lines = [
        "# CellBench (Tian et al. 2019)",
        "",
        "Generated by `scripts/cellbench.py`. RNA mixtures: pseudo-cells whose RNA "
        "is 100% one line (expected contamination 0) or 68/16/16 (expected 0.32). "
        "*gap* is the median of the mixtures minus that of the pure pseudo-cells: "
        "0.32 if the estimates follow the design. The paper's medians are from "
        "DecontX as published.",
        "",
        *table(
            mix,
            [
                ("n", "pseudo-cells", "{:d}"),
                ("pearson_design", "r with design", "{:.3f}"),
                ("median_pure", "median, pure", "{:.3f}"),
                ("median_68", "median, 68/16/16", "{:.3f}"),
                ("gap", "gap", "{:.3f}"),
                ("paper_median_pure", "paper, pure", "{:.3f}"),
                ("paper_median_68", "paper, 68/16/16", "{:.3f}"),
            ],
        ),
        "",
        "Single cells: median contamination of the singlets, and the AUROC with "
        "which contamination ranks demuxlet's doublets above the singlets.",
        "",
        *table(
            sc,
            [
                ("n", "cells", "{:d}"),
                ("n_doublets", "doublets", "{:d}"),
                ("median_singlets", "median", "{:.3f}"),
                ("paper_median", "paper", "{:.3f}"),
                ("doublet_auroc", "doublet AUROC", "{:.3f}"),
            ],
        ),
    ]
    (RESULTS / "cellbench.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
