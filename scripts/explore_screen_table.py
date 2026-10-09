"""
Quick exploratory analysis at glance of one or more screening (output of QC screening)
One table per dataset or per imageCAS group (in the test context), so new data can be judged rapidly.

Columns are found by their prefix, so the script keeps working when columns are added:
  spacing_*, z_extent_mm, fov_xy_mm   image homogeneity
  vol_*_ml                            volumes
  hu_*, noise_sd_hu                   intensities
  cnr_*, dhu_*                        right-heart contrast
  cut_*                               cut by the scan border
  wall_dist_*                         free-wall boundary position
  phase_*                             phase proxies
If a selection.csv (produced by earlier script) sits next to a table, included / excluded cases are marked and the
exclusion reasons are counted.

Robust outliers (|value - median| > 3 robust SD within each table) are labelled with
their case id, so odd cases can be opened in Slicer straight away for visualization.

Output: <out_dir>/explore_<name>.pdf (all pages) + one PNG per page.
Usage:  python explore_screen_table.py C:/data/rh_test/screen_table.csv [more tables] --out_dir ..
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # one colour per table (max 3 tables)
DIVERGING = "RdBu_r"                                 # correlation: red +, blue -, white 0
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
                     "axes.titlesize": 9, "axes.titleweight": "bold", "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})
ROBUST_Z = 3.0


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = 1.4826 * (x - med).abs().median()
    return (x - med) / mad if mad > 0 else x * 0.0


def load_tables(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["table"] = p.parent.name if p.stem == "screen_table" else p.stem
        sel = p.parent / "selection.csv"
        if sel.exists():
            s = pd.read_csv(sel)[["case", "included", "reasons"]]
            d = d.merge(s, on="case", how="left")
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    if "included" not in d:
        d["included"] = np.nan
    return d


def cols(d: pd.DataFrame, prefix: str, suffix: str = "") -> list[str]:
    return [c for c in d.columns if c.startswith(prefix) and c.endswith(suffix)
            and pd.api.types.is_numeric_dtype(d[c])]


def strip(ax, d: pd.DataFrame, columns: list[str], ylabel: str, title: str) -> None:
    """One column of dots per variable, one colour per table, median bar, outliers labelled."""
    tables = list(d.table.unique())
    width = 0.7 / max(len(tables), 1)
    for i, col in enumerate(columns):
        for k, t in enumerate(tables):
            sub = d[d.table == t]
            x0 = i - 0.35 + width * (k + 0.5)
            jitter = (np.random.default_rng(i * 10 + k).random(len(sub)) - 0.5) * width * 0.8
            excluded = sub.included == False                              # noqa: E712 (NaN-safe)
            ax.scatter(x0 + jitter[~excluded.values], sub[col][~excluded], s=12, color=SERIES[k % 3],
                       edgecolor=SURFACE, linewidth=0.5, zorder=3)
            ax.scatter(x0 + jitter[excluded.values], sub[col][excluded], s=14, marker="x",
                       color=INK2, linewidth=0.9, zorder=3)
            ax.plot([x0 - width * 0.4, x0 + width * 0.4], [sub[col].median()] * 2, color=INK, lw=1.8, zorder=4)
            z = robust_z(sub[col])
            for xj, (_, r) in zip(x0 + jitter[(z.abs() > ROBUST_Z).values], sub[z.abs() > ROBUST_Z].iterrows()):
                ax.annotate(r.case.replace("cas_", ""), (xj, r[col]), xytext=(3, 2),
                            textcoords="offset points", fontsize=6, color=INK2)
    ax.set_xticks(range(len(columns)), [c.split("_", 1)[1].replace("_ml", "").replace("_mm", "")
                                        for c in columns], rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def scatter(ax, d: pd.DataFrame, x: str, y: str, title: str) -> None:
    for k, t in enumerate(d.table.unique()):
        sub = d[d.table == t]
        excluded = sub.included == False                                  # noqa: E712
        ax.scatter(sub[x][~excluded], sub[y][~excluded], s=16, color=SERIES[k % 3], edgecolor=SURFACE,
                   linewidth=0.5, label=t, zorder=3)
        ax.scatter(sub[x][excluded], sub[y][excluded], s=18, marker="x", color=INK2, linewidth=0.9, zorder=3)
        out = (robust_z(sub[x]).abs() > ROBUST_Z) | (robust_z(sub[y]).abs() > ROBUST_Z)
        for _, r in sub[out].iterrows():
            ax.annotate(r.case.replace("cas_", ""), (r[x], r[y]), xytext=(3, 2),
                        textcoords="offset points", fontsize=6, color=INK2)
    r = d[[x, y]].corr(method="spearman").iloc[0, 1]
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{title}  (Spearman r = {r:.2f})")


def legend_note(fig, d: pd.DataFrame) -> None:
    tables = list(d.table.unique())
    handles = [plt.Line2D([], [], marker="o", ls="", color=SERIES[k % 3], label=f"{t} (n={int((d.table == t).sum())})")
               for k, t in enumerate(tables)]
    if d.included.notna().any():
        handles.append(plt.Line2D([], [], marker="x", ls="", color=INK2, label="excluded by selection"))
    handles.append(plt.Line2D([], [], color=INK, lw=1.8, label="median"))
    fig.legend(handles=handles, loc="upper right", frameon=False, ncol=len(handles), fontsize=7)


def page_image(d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    strip(axes[0], d, cols(d, "spacing_"), "mm", "Voxel spacing")
    strip(axes[1], d, [c for c in ("z_extent_mm", "fov_xy_mm") if c in d], "mm", "Scan coverage")
    axes[1].set_xticks([0, 1], ["z extent", "in-plane FOV"])
    if "direction" in d:
        txt = "\n".join(f"{v}: {n}" for v, n in d.direction.value_counts().items())
        axes[1].text(1.02, 0.02, "direction matrices\n" + txt, transform=axes[1].transAxes, fontsize=7, color=INK2)
    return fig, "Image homogeneity"


def page_volumes_hu(d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    strip(axes[0], d, cols(d, "vol_", "_ml"), "volume (mL)", "Volumes")
    strip(axes[1], d, cols(d, "hu_") + [c for c in ("noise_sd_hu",) if c in d], "HU",
          "Median HU per structure, and noise")
    return fig, "Volumes and intensities"


def page_contrast(d):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    if {"cnr_RA", "cnr_RV"} <= set(d.columns):
        scatter(axes[0], d, "cnr_RA", "cnr_RV", "Contrast-to-noise vs LV wall")
        for v in (0.5, 2.0):
            axes[0].axhline(v, color=MUTED, ls="--", lw=0.8)
            axes[0].axvline(v, color=MUTED, ls="--", lw=0.8)
    if {"wall_dist_RA_mm", "wall_dist_RV_mm"} <= set(d.columns):
        scatter(axes[1], d, "wall_dist_RA_mm", "wall_dist_RV_mm", "Free-wall boundary to fat/lung")
    cut = cols(d, "cut_") or [c for c in d.columns if c.startswith("cut_")]
    if cut:
        tables = list(d.table.unique())
        width = 0.8 / len(tables)
        for k, t in enumerate(tables):
            sub = d[d.table == t]
            frac = [100 * sub[c].astype(bool).mean() for c in cut]
            axes[2].bar(np.arange(len(cut)) + (k - (len(tables) - 1) / 2) * width, frac, width * 0.9,
                        color=SERIES[k % 3])
        axes[2].set_xticks(range(len(cut)), [c.replace("cut_", "") for c in cut])
        axes[2].set_ylabel("% of cases cut by the scan border")
        axes[2].set_title("Completeness")
    return fig, "Contrast, boundary position, completeness"


def page_phase(d):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if {"phase_la_over_lv", "phase_ra_over_rv"} <= set(d.columns):
        scatter(axes[0], d, "phase_la_over_lv", "phase_ra_over_rv", "Phase proxies")
    ph = cols(d, "phase_")
    for k, t in enumerate(d.table.unique()):
        for j, c in enumerate(ph):
            axes[1].hist(d.loc[d.table == t, c].dropna(), bins=20, histtype="step", lw=1.4,
                         color=SERIES[k % 3], linestyle=["-", "--", ":"][j % 3],
                         label=f"{c.replace('phase_', '')} ({t})")
    axes[1].set_xlabel("ratio")
    axes[1].set_ylabel("cases")
    axes[1].set_title("Phase proxy distributions (two peaks would mean two phases)")
    axes[1].legend(frameon=False, fontsize=6)
    return fig, "Cardiac phase"


def page_correlation(d):
    num = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and d[c].nunique() > 2
           and not c.startswith("spacing_")]
    corr = d[num].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(num)), num, rotation=90, fontsize=6)
    ax.set_yticks(range(len(num)), num, fontsize=6)
    ax.grid(False)
    for i in range(len(num)):
        for j in range(len(num)):
            if i != j and abs(corr.values[i, j]) >= 0.5:
                ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center", va="center", fontsize=5, color=INK)
    fig.colorbar(im, ax=ax, fraction=0.03, label="Spearman correlation")
    ax.set_title("Which indices move together (|r| >= 0.5 printed)")
    return fig, "Correlations"


def page_selection(d):
    if d.included.isna().all():
        return None, None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    reasons = d.reasons.fillna("").str.split(";").explode()
    counts = reasons[reasons != ""].value_counts().sort_values()
    axes[0].barh(counts.index, counts.values, color=SERIES[0], height=0.6)
    for y, n in enumerate(counts.values):
        axes[0].text(n + 0.3, y, str(n), va="center", fontsize=7, color=INK)
    axes[0].grid(axis="y", visible=False)
    axes[0].set_xlabel("cases failing this check (a case can fail several)")
    axes[0].set_title("Exclusion reasons")
    tables = list(d.table.unique())
    n_all = [int((d.table == t).sum()) for t in tables]
    has_sel = [d.loc[d.table == t, "included"].notna().any() for t in tables]
    n_in = [int(d.loc[d.table == t, "included"].fillna(False).astype(bool).sum()) for t in tables]
    axes[1].bar(tables, n_all, color=GRID, label="screened")
    axes[1].bar(tables, n_in, color=SERIES[0], label="included")
    for i, (a, b, s) in enumerate(zip(n_in, n_all, has_sel)):
        axes[1].text(i, b + 0.5, f"{a}/{b}" if s else "no selection.csv", ha="center", fontsize=8, color=INK)
    axes[1].set_ylabel("cases")
    axes[1].set_title("Included per table")
    axes[1].legend(frameon=False)
    return fig, "Selection"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", type=Path, nargs="+", help="one or more screen_table.csv files")
    parser.add_argument("--out_dir", type=Path, default=None, help="default: folder of the first table / qc")
    parser.add_argument("--name", default="screening")
    args = parser.parse_args()
    out_dir = args.out_dir or args.tables[0].parent / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_tables(args.tables)
    print(f"{len(d)} cases from {d.table.nunique()} table(s): {', '.join(d.table.unique())}")
    with PdfPages(out_dir / f"explore_{args.name}.pdf") as pdf:
        for i, page in enumerate((page_image, page_volumes_hu, page_contrast, page_phase,
                                  page_correlation, page_selection), 1):
            fig, title = page(d)
            if fig is None:
                continue
            fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
            legend_note(fig, d)
            fig.tight_layout(rect=(0, 0, 1, 0.93))
            fig.savefig(out_dir / f"explore_{args.name}_{i}.png", dpi=140)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"-> {out_dir / f'explore_{args.name}.pdf'} (+ PNG per page)")