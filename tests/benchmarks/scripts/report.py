"""Make the tables and figures of the benchmark from results/.

Usage: python scripts/report.py

Reads results/results.json and results/cells/, and writes results/metrics.csv,
results/tables.md and figures/{accuracy,speed,accuracy_vs_time}.png.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

NAMES = {
    "scdecontx": "scdecontx",
    "R": "R celda",
    "niruff": "NiRuff",
    "omicverse": "omicverse",
}
COLORS = {
    "scdecontx": "#2a78d6",
    "R": "#eb6834",
    "niruff": "#1baf7a",
    "omicverse": "#eda100",
}
MARKERS = {"scdecontx": "o", "R": "s", "niruff": "v", "omicverse": "P"}
TOOLS = ["scdecontx", "R", "niruff", "omicverse"]
GROUP_TITLES = {
    "sweep_g15000": "3,000 cells × 15,000 genes × 8 clusters",
    "sweep_g1000": "2,000 cells × 1,000 genes × 5 clusters",
    "merged": "Merged clusters: 3,000 cells × 15,000 genes, 16 types as 8 clusters",
}
INK, INK2, GRID = "#1f1f1e", "#5e5d59", "#e8e7e2"
SECONDS = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20]


# Data ---------------------------------------------------------------------


def tidy(ds) -> pd.DataFrame:
    """One row for each dataset and tool run."""
    rows = []
    for name, d in ds.items():
        base = {
            k: d.get(k)
            for k in ("group", "n_cells", "n_genes", "nnz", "n_clusters", "true_mean")
        }
        base["dataset"] = name
        for tool, e in d.items():
            if not isinstance(e, dict):
                continue
            is_r = tool.startswith("R")
            times = e.get("em_s" if is_r else "warm_s") or []
            row = {
                **base,
                "tool": tool,
                "mae": e.get("true_contam_mae"),
                "bias": e.get("true_contam_bias"),
                "pearson": e.get("true_contam_pearson"),
                "counts_rel_error": e.get("true_native_relfro"),
                "mean_contamination": e.get("mean_contamination"),
                "self_consistency": e.get("self_consistency"),
                "nonzeros_kept": e.get("frac_nonzero_kept"),
                "r_contam_maxdiff": e.get("r_contam_maxdiff"),
                "r_counts_maxdiff": e.get("r_counts_maxdiff"),
                "n_iter": e.get("n_iter"),
                "delta_native": (e.get("delta") or [None, None])[0],
                "delta_contamination": (e.get("delta") or [None, None])[1],
                "time_median_s": float(np.median(times)) if times else None,
                "time_min_s": float(np.min(times)) if times else None,
                "time_max_s": float(np.max(times)) if times else None,
                "n_timed": int(np.size(times)),
                "time_total_median_s": float(np.median(e["total_s"])) if is_r else None,
                "time_cold_s": e.get("cold_s"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.1e}"
    return f"{x:.{digits}g}"


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def level_of(df):
    """Contamination level (rounded true mean) of each sweep dataset."""
    return df.groupby("dataset")["true_mean"].transform("first").round(1)


# Tables -------------------------------------------------------------------


def speed_names(m, n_genes=None):
    """Speed datasets, smallest first. If n_genes is given, keep only those."""
    speed = m[m.group == "speed"].drop_duplicates("dataset")
    if n_genes is not None:
        speed = speed[speed.n_genes == n_genes]
    return list(speed.sort_values(["n_genes", "n_cells"]).dataset)


def summary_table(m):
    def mae(group, tool):
        sel = m[(m.group == group) & (m.tool == tool)]
        return sel.mae.mean(), sel.mae.std()

    def bold_min(vals):
        low = min(v for v, _ in vals)
        return [
            ("**" if v == low else "")
            + f"{v:.3f} ± {s:.3f}"
            + ("**" if v == low else "")
            for v, s in vals
        ]

    rows = []
    for group, label in [
        ("sweep_g15000", "Contamination MAE, 15,000 genes (n = 15)"),
        ("sweep_g1000", "Contamination MAE, 1,000 genes (n = 15)"),
        ("merged", "Contamination MAE, merged clusters (n = 4)"),
    ]:
        rows.append([label, *bold_min([mae(group, t) for t in TOOLS])])
    sel = m[m.group == "sweep_g15000"]
    vals = [
        (
            sel[sel.tool == t].counts_rel_error.mean(),
            sel[sel.tool == t].counts_rel_error.std(),
        )
        for t in TOOLS
    ]
    rows.append(["Counts relative error, 15,000 genes (n = 15)", *bold_min(vals)])
    pb = m[m.dataset == "pbmc3k"].set_index("tool")
    rows.append(
        [
            "Run time, PBMC3k (s)",
            fmt(pb.time_median_s["scdecontx"]),
            f"{fmt(pb.time_total_median_s['R'])} ({fmt(pb.time_median_s['R'])} EM)",
            *[fmt(pb.time_median_s[t]) for t in ["niruff", "omicverse"]],
        ]
    )
    timed = m[m.group.isin(["speed", "real"])]
    ratios = []
    for t in ["R", "niruff", "omicverse"]:
        r = (
            timed[timed.tool == t].set_index("dataset").time_median_s
            / timed[timed.tool == "scdecontx"].set_index("dataset").time_median_s
        )
        ratios.append(f"{r.min():.1f}–{r.max():.1f}× slower")
    rows.append(
        [
            "Run time vs scdecontx, same stop rule",
            "1×",
            ratios[0] + " (EM only)",
            *ratios[1:],
        ]
    )
    cons = m.groupby("tool").self_consistency.max()
    rows.append(
        ["Max contamination − (1 − counts kept)", *[fmt(cons[t]) for t in TOOLS]]
    )
    kept = m[m.tool == "niruff"].nonzeros_kept
    rows.append(
        [
            "Input nonzeros kept",
            "100%",
            "100%",
            f"{kept.min():.0%}–{kept.max():.0%}",
            "100%",
        ]
    )
    shared = m[m.tool.str.endswith("_shared")].groupby("tool").r_contam_maxdiff.max()
    rows.append(
        [
            "Max contamination difference from R, same start",
            fmt(shared["scdecontx_shared"]),
            "–",
            fmt(shared["niruff_shared"]),
            fmt(shared["omicverse_shared"]),
        ]
    )
    return table(["", *[NAMES[t] for t in TOOLS]], rows)


def sweep_table(m, group):
    sel = m[m.group == group].copy()
    sel["level"] = level_of(sel)
    rows = []
    for level, lv in sel.groupby("level"):
        stats = [
            (lv[lv.tool == t].mae.mean(), lv[lv.tool == t].mae.std()) for t in TOOLS
        ]
        low = min(v for v, _ in stats)
        cells = [
            ("**" if v == low else "")
            + f"{fmt(v)} ± {fmt(s, 2)}"
            + ("**" if v == low else "")
            for v, s in stats
        ]
        bias = lv[lv.tool == "scdecontx"].bias.mean()
        rows.append([f"{level:.0%}", str(lv.dataset.nunique()), *cells, f"{bias:+.3f}"])
    return table(
        ["True contamination", "n", *[NAMES[t] for t in TOOLS], "Mean bias"],
        rows,
    )


def timed_calls(m):
    """Describe the number of timed calls behind Table 3, from the results."""
    names = [*speed_names(m), "pbmc3k"]
    n = m[m.dataset.isin(names) & (m.n_timed > 0)].groupby("dataset").n_timed.max()
    common = int(n.mode().iloc[0])
    others = {int(k): sorted(n.index[n == k]) for k in n.unique() if k != common}
    text = f"{common} warm calls"
    if others:
        text += (
            " ("
            + "; ".join(f"{k} for {', '.join(v)}" for k, v in sorted(others.items()))
            + ")"
        )
    return text


def speed_table(m):
    rows = []
    for name in [*speed_names(m), "pbmc3k"]:
        d = m[m.dataset == name].set_index("tool")

        def t(tool, d=d):
            if tool not in d.index:
                return "–"
            r = d.loc[tool]
            return f"{fmt(r.time_median_s)} ({fmt(r.time_min_s)}–{fmt(r.time_max_s)})"

        rows.append(
            [
                name,
                f"{int(d.n_cells.iloc[0]):,}",
                f"{int(d.n_genes.iloc[0]):,}",
                f"{int(d.nnz.iloc[0]):,}",
                fmt(d.time_total_median_s["R"]),
                t("R"),
                t("scdecontx"),
                t("scdecontx_1t"),
                t("niruff"),
                t("omicverse"),
            ]
        )
    return table(
        [
            "Dataset",
            "Cells",
            "Genes",
            "Nonzeros",
            "R total",
            "R EM",
            "scdecontx",
            "scdecontx, 1 thread",
            "NiRuff",
            "omicverse",
        ],
        rows,
    )


def iterations_table(m):
    rows = []
    for name in [*speed_names(m), "pbmc3k"]:
        d = m[m.dataset == name].set_index("tool").n_iter
        rows.append([name, *[fmt(d.get(t)) for t in ["R", "scdecontx", "omicverse"]]])
    return table(["Dataset", "R", "scdecontx", "omicverse"], rows)


def agreement_table(m):
    rows = []
    for name in m[m.group.isin(["accuracy", "real"])].dataset.unique():
        d = m[m.dataset == name].set_index("tool")
        row = [name]
        for t in ["scdecontx", "niruff", "omicverse"]:
            for suffix in ("", "_shared"):
                r = d.loc[t + suffix]
                row.append(f"{fmt(r.r_contam_maxdiff)} / {fmt(r.r_counts_maxdiff)}")
        rows.append(row)
    return table(
        [
            "Dataset",
            "scdecontx, own seed",
            "scdecontx, same start",
            "NiRuff, own seed",
            "NiRuff, same start",
            "omicverse, own seed",
            "omicverse, same start",
        ],
        rows,
    )


def pbmc_table():
    cells = pd.read_csv(RESULTS / "cells" / "pbmc3k.csv.gz")
    rows = []
    for k, c in cells.groupby("cluster"):
        rows.append([str(k), str(len(c)), *[f"{c[t].mean():.1%}" for t in TOOLS]])
    rows.append(
        ["all", str(len(cells)), *[f"**{cells[t].mean():.1%}**" for t in TOOLS]]
    )
    return table(["Cluster", "Cells", *[NAMES[t] for t in TOOLS]], rows)


def integrity_table(m):
    rows = []
    for name in m[m.group.isin(["accuracy", "real", "speed"])].dataset.unique():
        d = m[m.dataset == name].set_index("tool")
        rows.append(
            [name]
            + [fmt(d.self_consistency[t]) for t in TOOLS]
            + [f"{d.nonzeros_kept[t]:.1%}" for t in TOOLS]
        )
    return table(
        [
            "Dataset",
            *[f"{NAMES[t]} consistency" for t in TOOLS],
            *[f"{NAMES[t]} nonzeros kept" for t in TOOLS],
        ],
        rows,
    )


def write_tables(m):
    w = []
    w.append("# Benchmark tables\n")
    w.append(
        "Generated by `scripts/report.py` from `results/results.json`. Values are mean ± SD over "
        "datasets (seeds) unless marked; n is the number of datasets. MAE is the mean absolute "
        "error of per-cell contamination against the simulated truth. The lowest value in each "
        "row is bold. scdecontx runs celda's algorithm, so its estimates equal R's from the "
        "same start; its rows differ from R's only through the random initial values. See "
        "`README.md` for the design.\n"
    )
    w.append("## Table 1. Summary, each tool at its defaults\n")
    w.append(summary_table(m))
    w.append("\n## Table 2. Contamination MAE by contamination level\n")
    w.append(
        "Bias is the mean signed error of scdecontx; negative means underestimated.\n"
    )
    for group in ["sweep_g15000", "sweep_g1000", "merged"]:
        w.append(f"**{GROUP_TITLES[group]}**\n")
        w.append(sweep_table(m, group))
        w.append("")
    w.append("## Table 3. Run time (s)\n")
    w.append(
        f"Median (min–max) of {timed_calls(m)} in a fresh process, each tool at its default stop "
        "rule (all: tol 1e-3, at most 500 iterations). *R total* includes the UMAP that celda "
        "computes in every call; *R EM* excludes it. scdecontx uses 11 threads; the 1-thread "
        "column is single-threaded like R.\n"
    )
    w.append(speed_table(m))
    w.append("\nIterations to convergence:\n")
    w.append(iterations_table(m))
    w.append("\n## Table 4. Agreement with R celda\n")
    w.append(
        "Max absolute difference from R in per-cell contamination / in any decontaminated "
        "count. *Same start* runs begin from R's own initial theta, captured with `trace()`.\n"
    )
    w.append(agreement_table(m))
    w.append("\n## Table 5. PBMC3k: mean contamination by cluster\n")
    w.append(
        "Real data without ground truth; clusters are Leiden clusters (resolution 0.5).\n"
    )
    w.append(pbmc_table())
    w.append("\n## Table 6. Output integrity\n")
    w.append(
        "*Consistency*: max over cells of |contamination − (1 − decontaminated total / "
        "observed total)|, 0 when the reported contamination describes the returned counts. "
        "*Nonzeros kept*: fraction of input nonzeros still nonzero in the output.\n"
    )
    w.append(integrity_table(m))
    w.append("")
    (RESULTS / "tables.md").write_text("\n".join(w) + "\n")


# Figures ------------------------------------------------------------------


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")


def fig_accuracy(m, plt):
    """MAE of each implementation by true contamination level.
    One panel per data group."""
    tools = ["R", "omicverse", "niruff", "scdecontx"]
    groups = ["sweep_g15000", "sweep_g1000", "merged"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, group, letter in zip(axes, groups, "abc", strict=True):
        sel = m[m.group == group].copy()
        sel["level"] = level_of(sel)
        for tool in tools:
            st = sel[sel.tool == tool].groupby("level").mae.agg(["mean", "std"])
            ax.errorbar(
                st.index * 100,
                st["mean"],
                yerr=st["std"],
                color=COLORS[tool],
                marker=MARKERS[tool],
                ms=6,
                lw=1.8,
                capsize=2,
                label=NAMES[tool],
                markeredgecolor="white",
                markeredgewidth=0.8,
            )
        ax.set_title(
            f"{letter}  {GROUP_TITLES[group].split(':')[-1].strip()}",
            loc="left",
            fontsize=9,
            color=INK,
        )
        ax.set_xlabel("True mean contamination (%)", fontsize=8)
        ax.set_ylim(bottom=0)
        style(ax)
    axes[0].set_ylabel("Contamination MAE (lower is better)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.06),
    )
    fig.tight_layout()
    save(fig, "accuracy")


def fig_speed(m, plt):
    """Wall time of one call against the number of cells.
    Each tool at its default settings."""
    tools = ["omicverse", "R", "scdecontx", "niruff"]
    names = speed_names(m, n_genes=2000)
    sel = m[m.dataset.isin(names)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    r = sel[sel.tool == "R"].set_index("dataset").loc[names]
    ax.plot(
        r.n_cells,
        r.time_total_median_s,
        color=COLORS["R"],
        ls="--",
        lw=1.4,
        marker=MARKERS["R"],
        ms=5,
        markerfacecolor="white",
    )
    ax.annotate(
        "R celda, full call (with UMAP)",
        (r.n_cells.iloc[-1], r.time_total_median_s.iloc[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK,
    )
    for tool in tools:
        d = sel[sel.tool == tool].set_index("dataset").loc[names]
        ax.plot(
            d.n_cells,
            d.time_median_s,
            color=COLORS[tool],
            marker=MARKERS[tool],
            ms=6,
            lw=1.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        label = NAMES[tool] + (", EM only" if tool == "R" else "")
        ax.annotate(
            label,
            (d.n_cells.iloc[-1], d.time_median_s.iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    cells = sel.n_cells.drop_duplicates().sort_values()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(cells, [f"{c // 1000},000" for c in cells])
    ax.set_yticks(SECONDS, [f"{v:g} s" for v in SECONDS])
    ax.minorticks_off()
    ax.set_xlim(cells.iloc[0] * 0.8, cells.iloc[-1] * 4)
    ax.set_xlabel("Cells (2,000 genes, 8 clusters)", fontsize=8)
    ax.set_ylabel(
        "Wall time per call (seconds, median of 3; lower is better)", fontsize=8
    )
    style(ax)
    fig.tight_layout()
    save(fig, "speed")


def fig_tradeoff(m, plt):
    """Run time and error as two bar panels (journal single column, 89 mm, Arial)."""
    tools = ["scdecontx", "R", "omicverse", "niruff"]
    labels = {
        "scdecontx": "scdecontx",
        "R": "R celda",
        "omicverse": "omicverse",
        "niruff": "NiRuff",
    }
    bar_colors = {
        "scdecontx": "#2a78d6",
        "R": "#eb6834",
        "omicverse": "#eda100",
        "niruff": "#1baf7a",
    }
    mae = m[(m.group == "sweep_g15000") & m.tool.isin(tools)].groupby("tool").mae.mean()
    big = "speed_20k_g15k"
    t = m[(m.dataset == big) & m.tool.isin(tools)].set_index("tool")
    secs = t.time_median_s.copy()
    secs["R"] = t.loc["R", "time_total_median_s"]
    # Sort by the lowest error first. Break ties by run time.
    tools = sorted(tools, key=lambda tool: (round(mae[tool], 3), secs[tool]))
    mm = 1 / 25.4
    with plt.rc_context(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 0,
            "xtick.direction": "out",
        }
    ):
        fig = plt.figure(figsize=(89 * mm, 50 * mm))
        grid = fig.add_gridspec(
            1,
            2,
            hspace=0.55,
            wspace=0.3,
            bottom=0.3,
            top=0.86,
            left=0.06,
            right=0.98,
        )
        ax_a = fig.add_subplot(grid[0, 0])
        ax_b = fig.add_subplot(grid[0, 1], sharey=ax_a)
        axes = [ax_a, ax_b]
        plt.setp(ax_b.get_yticklabels(), visible=False)
        y = range(len(tools))[::-1]
        panels = [
            (
                axes[0],
                "a",
                "Run time (s)\n20,000 cells \u00d7 15,000 genes",
                secs,
                "{:.2g} s",
                float(secs.max()) * 1.3,
            ),
            (
                axes[1],
                "b",
                "Error (MAE)\n3,000 cells \u00d7 15,000 genes",
                mae,
                "{:.3f}",
                0.13,
            ),
        ]
        for ax, letter, title, values, fmt, xmax in panels:
            vals = [values[tool] for tool in tools]
            ax.barh(
                list(y),
                vals,
                height=0.62,
                color=[bar_colors[tool] for tool in tools],
                lw=0,
            )
            for yi, v in zip(y, vals, strict=True):
                ax.text(
                    v + xmax * 0.02,
                    yi,
                    fmt.format(v),
                    va="center",
                    fontsize=6.5,
                    color=INK,
                )
            ax.set_xlim(0, xmax)
            ax.set_title(
                f"{letter}  {title}", loc="left", fontsize=6.5, fontweight="bold", pad=4
            )
            ax.tick_params(labelsize=6.5, colors=INK)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(INK)
        axes[0].set_yticks([])
        axes[0].xaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
        axes[1].set_xticks([0, 0.05, 0.1])
        fig.legend(
            handles=[plt.Rectangle((0, 0), 1, 1, color=bar_colors[t_]) for t_ in tools],
            labels=[labels[t_] for t_ in tools],
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=6,
            handlelength=1,
            handleheight=0.8,
            bbox_to_anchor=(0.5, -0.01),
        )
        save(fig, "accuracy_vs_time")


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "sans-serif"})
    res = json.loads((RESULTS / "results.json").read_text())
    m = tidy(res["datasets"])
    m.to_csv(RESULTS / "metrics.csv", index=False, float_format="%.6g")
    write_tables(m)
    FIGURES.mkdir(exist_ok=True)
    fig_accuracy(m, plt)
    fig_speed(m, plt)
    fig_tradeoff(m, plt)
    print("wrote", RESULTS / "metrics.csv", RESULTS / "tables.md", FIGURES)


if __name__ == "__main__":
    main()
