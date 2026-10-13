"""
High-level review of the cohort after merging and opening cut: one table with every number from te merge reports and orifices descriptors
plus plots that make bad anatomies stand out.

Per case, in cohort_qc.csv:
  merge    volumes, mean HU, % of a vessel lost to the chambers, islands removed, holes filled,
           vessel-chamber gap and contact
  orifices equivalent diameter, long/short axis ratio and fill depth of the SVC, IVC and RVOT
           openings, tilt of the valve plane against the PA trunk, distances between the
           opening centres and angles between their normals (a plausible right heart has
           them in a plausible layout)
  surface  (if 06 has run) mesh quality and cap flatness

A case is listed in review_cases.csv when it has a failed check or when any index lies more
than FLAG_Z robust standard deviations from the cohort median, with the reasons written out.

Output (in --out_dir): cohort_qc.csv, review_cases.csv, cohort_qc.pdf + one PNG per page.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import numpy as np

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})

LABELS = ("RA", "RV", "PA", "SVC", "IVC")
OPENINGS = ("SVC", "IVC", "RVOT")
FLAG_Z = 3.5                        # robust z beyond which an index flags the case
# smallest spread (in the index's own unit) that counts as meaningful variation. Without it, an
# index that is almost always 0 (e.g. % of the SVC given to the RA) would flag any non-zero value.
MIN_SCALE = {"vol_": 5.0, "hu_": 10.0, "lost_": 3.0, "gap_": 0.1, "diam_": 1.0, "ratio_": 0.1,
             "fill_depth_": 0.5, "valve_tilt": 3.0, "dist_": 2.0, "angle_": 5.0}
# indices that can flag a case, with a readable name
FLAG_INDICES = {
    "vol_RA_ml": "RA volume", "vol_RV_ml": "RV volume", "hu_RA": "RA HU", "hu_RV": "RV HU",
    "lost_SVC_pct": "SVC lost to RA", "lost_IVC_pct": "IVC lost to RA", "gap_SVC-RA_mm": "SVC-RA gap",
    "gap_IVC-RA_mm": "IVC-RA gap", "gap_PA-RV_mm": "PA-RV gap",
    "diam_SVC_mm": "SVC opening diameter", "diam_IVC_mm": "IVC opening diameter",
    "diam_RVOT_mm": "RVOT opening diameter", "ratio_SVC": "SVC axis ratio", "ratio_IVC": "IVC axis ratio",
    "ratio_RVOT": "RVOT axis ratio", "fill_depth_SVC_mm": "SVC fill depth", "fill_depth_IVC_mm": "IVC fill depth",
    "fill_depth_RVOT_mm": "RVOT fill depth", "valve_tilt_deg": "valve-plane tilt",
    "dist_SVC_IVC_mm": "SVC-IVC distance", "dist_SVC_RVOT_mm": "SVC-RVOT distance",
    "dist_IVC_RVOT_mm": "IVC-RVOT distance"}

# ---------------------------------------------------------------- 1. read the reports
def read_case(case_dir: Path) -> dict:
    merge = json.loads((case_dir / "merge_report.json").read_text())
    orifices = json.loads((case_dir / "orifices.json").read_text())
    row = {"case": case_dir.name}

    for lab in LABELS:
        row[f"vol_{lab}_ml"] = merge[lab]["volume_ml"]
        row[f"hu_{lab}"] = merge[lab]["mean_hu"]
        row[f"lost_{lab}_pct"] = 100.0 * merge[lab]["lost_to_higher_priority"] / max(merge[lab]["voxels_raw"], 1)
    row["islands_removed_vox"] = sum(merge["removed_islands_voxels"].values())
    row["filled_cavity_vox"] = merge["filled_cavity_voxels"]
    for j in merge["junctions"]:
        row[f"gap_{j['junction']}_mm"] = j.get("gap_mm", np.nan)
        row[f"contact_{j['junction']}_vox"] = j.get("contact_voxels", np.nan)
    row["merge_problems"] = ";".join(merge["problems"])

    centres, normals = {}, {}
    for name in OPENINGS:
        o = orifices[name]
        if "semi_axes_mm" not in o:                       # the opening could not be built: only a status exists
            row[f"status_{name}"] = o.get("status", "missing")
            continue
        a, b = o["semi_axes_mm"]
        row[f"diam_{name}_mm"] = o["equivalent_diameter_mm"]
        row[f"ratio_{name}"] = a / b
        row[f"fill_depth_{name}_mm"] = o["recess_filled_ml"] * 1000.0 / (np.pi * a * b)   # mean fill depth, same formula as 05
        row[f"status_{name}"] = o["status"]
        centres[name], normals[name] = np.array(o["centre_world"]), np.array(o["normal_world"])
    row["valve_tilt_deg"] = orifices["RVOT"].get("angle_valve_plane_to_trunk_deg", np.nan)
    for a, b in (("SVC", "IVC"), ("SVC", "RVOT"), ("IVC", "RVOT")):
        if a in centres and b in centres:
            row[f"dist_{a}_{b}_mm"] = float(np.linalg.norm(centres[a] - centres[b]))
            cos = np.clip(normals[a] @ normals[b], -1, 1)
            row[f"angle_{a}_{b}_deg"] = float(np.degrees(np.arccos(cos)))
    row["orifice_problems"] = ";".join(orifices["problems"])

    surface = case_dir / "surface_qc.json"
    if surface.exists():
        s = json.loads(surface.read_text())
        row["surface_included"] = s["included"]
        if "vertices" not in s:                           # 06 stopped before meshing (an opening is missing)
            row["mesh_problems"] = ";".join(s["problems"])
            return row
        row.update({"mesh_vertices": s["vertices"], "mesh_edge_mm": s["edge_mm_mean"], "mesh_min_angle_deg": s["min_angle_deg"],
                    "mesh_max_valence": s["max_valence"], "mesh_volume_ml": s["volume_ml"],
                    "mesh_self_intersections": s["self_intersecting_faces"], "mesh_problems": ";".join(s["problems"])})
        for name, cap in s["caps"].items():
            row[f"cap_{name}_area_ratio"] = cap["area_ratio"]
            row[f"cap_{name}_plane_rms_mm"] = cap["plane_rms_mm"]
    return row

# ---------------------------------------------------------------- 2. flag cases
def robust_z(x: pd.Series, name: str = "") -> pd.Series:
    """(x - median) / spread, spread = 1.4826 x MAD, but never below the index's MIN_SCALE."""
    med = x.median()
    floor = next((v for k, v in MIN_SCALE.items() if name.startswith(k)), 0.0)
    spread = max(1.4826 * (x - med).abs().median(), floor, 1e-9)
    return (x - med) / spread


def flag_cases(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robust z-score matrix for the flagging indices, and the list of cases to review with reasons."""
    # .values: the z-scores are aligned by position; indexing them by case name would reindex them to NaN
    z = pd.DataFrame({c: robust_z(d[c], c).values for c in FLAG_INDICES if c in d}, index=d.case)
    reasons = {case: [] for case in d.case}
    for case, row in z.iterrows():
        for col, val in row.items():
            if abs(val) > FLAG_Z:
                reasons[case].append(f"{FLAG_INDICES[col]} z={val:+.1f}")
    for _, r in d.iterrows():
        for col in ("merge_problems", "orifice_problems", "mesh_problems"):
            if col in d and isinstance(r[col], str) and r[col]:
                reasons[r.case].append(f"{col.split('_')[0]} check failed: {r[col]}")
    review = pd.DataFrame([{"case": c, "n_flags": len(v), "reasons": "; ".join(v)} for c, v in reasons.items() if v])
    return z, review.sort_values("n_flags", ascending=False) if len(review) else review


# ---------------------------------------------------------------- 3. plots
def strip(ax, d, columns, ylabel, title, log=False):
    """One column of dots per index, median bar, outliers (robust z beyond FLAG_Z) labelled."""
    rng = np.random.default_rng(0)
    for i, col in enumerate(columns):
        if col not in d:
            continue
        v = d[col]
        x = i + (rng.random(len(d)) - 0.5) * 0.5
        out = robust_z(v, col).abs() > FLAG_Z
        ax.scatter(x[~out], v[~out], s=12, color=BLUE, edgecolor=SURFACE, linewidth=0.4, zorder=3)
        ax.scatter(x[out], v[out], s=22, color=ORANGE, edgecolor=SURFACE, linewidth=0.4, zorder=4)
        ax.plot([i - 0.3, i + 0.3], [v.median()] * 2, color=INK, lw=1.8, zorder=5)
        for xi, vi, case in zip(x[out], v[out], d.case[out]):
            ax.annotate(case.replace("cas_", ""), (xi, vi), xytext=(3, 2), textcoords="offset points", fontsize=6, color=INK2)
    ax.set_xticks(range(len(columns)), [c.replace("_ml", "").replace("_mm", "").replace("_pct", "") for c in columns],
                  rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(bottom=0)


def page(title, nrows, ncols, size):
    fig, axes = plt.subplots(nrows, ncols, figsize=size)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    return fig, np.atleast_1d(axes).ravel()


def scatter(ax, d, x, y, title):
    ax.scatter(d[x], d[y], s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.4, zorder=3)
    r = d[[x, y]].corr(method="spearman").iloc[0, 1]
    out = (robust_z(d[x], x).abs() > FLAG_Z) | (robust_z(d[y], y).abs() > FLAG_Z)
    ax.scatter(d.loc[out, x], d.loc[out, y], s=22, color=ORANGE, zorder=4)
    for _, r_ in d[out].iterrows():
        ax.annotate(r_.case.replace("cas_", ""), (r_[x], r_[y]), xytext=(3, 2), textcoords="offset points", fontsize=6, color=INK2)
    ax.set_xlabel(x.replace("_ml", " (mL)").replace("_mm", " (mm)"))
    ax.set_ylabel(y.replace("_mm", " (mm)"))
    ax.set_title(f"{title}   Spearman r = {r:.2f}")


def build_pages(d, z, review):
    pages = []
    fig, ax = page("Merge: volumes and intensities", 1, 2, (11, 4.6))
    strip(ax[0], d, [f"vol_{lab}_ml" for lab in LABELS], "mL", "Volumes after merge and cleanup")
    strip(ax[1], d, [f"hu_{lab}" for lab in LABELS], "HU", "Mean HU inside each structure")
    pages.append(fig)

    fig, ax = page("Merge: what cleanup changed, and how vessels meet chambers", 2, 2, (11, 8))
    strip(ax[0], d, [f"lost_{lab}_pct" for lab in ("SVC", "IVC")], "% of the vessel", "Vessel voxels given to the chambers")
    strip(ax[1], d, ["islands_removed_vox", "filled_cavity_vox"], "voxels", "Islands removed / holes filled", log=True)
    strip(ax[2], d, [c for c in d.columns if c.startswith("gap_")], "mm", "Vessel-to-chamber gap (0.3-0.5 = touching)")
    strip(ax[3], d, [c for c in d.columns if c.startswith("contact_")], "voxels", "Touching voxels (informational)", log=True)
    pages.append(fig)

    fig, ax = page("Openings: size and shape", 2, 2, (11, 8))
    strip(ax[0], d, [f"diam_{n}_mm" for n in OPENINGS], "mm", "Equivalent diameter")
    strip(ax[1], d, [f"ratio_{n}" for n in OPENINGS], "long / short axis", "Axis ratio (1 = round)")
    strip(ax[2], d, [f"fill_depth_{n}_mm" for n in OPENINGS], "mm", "Fill depth to make the opening flat (flag > 6)")
    strip(ax[3], d, ["valve_tilt_deg"], "degrees", "Tilt: valve plane vs PA trunk")
    pages.append(fig)

    fig, ax = page("Openings: layout of the three openings (a plausible right heart keeps a plausible layout)", 2, 3, (13, 8))
    strip(ax[0], d, [c for c in d.columns if c.startswith("dist_")], "mm", "Distances between opening centres")
    strip(ax[1], d, [c for c in d.columns if c.startswith("angle_")], "degrees", "Angles between opening normals")
    scatter(ax[2], d, "vol_RA_ml", "diam_SVC_mm", "RA volume vs SVC opening")
    scatter(ax[3], d, "vol_RA_ml", "diam_IVC_mm", "RA volume vs IVC opening")
    scatter(ax[4], d, "vol_RV_ml", "diam_RVOT_mm", "RV volume vs RVOT opening")
    scatter(ax[5], d, "dist_SVC_IVC_mm", "vol_RA_ml", "SVC-IVC distance vs RA volume")
    pages.append(fig)

    if "mesh_vertices" in d:
        fig, ax = page("Surface mesh (06)", 2, 3, (13, 8))
        strip(ax[0], d, ["mesh_edge_mm"], "mm", "Mean edge length")
        strip(ax[1], d, ["mesh_min_angle_deg"], "degrees", "Smallest triangle angle")
        strip(ax[2], d, ["mesh_max_valence"], "edges at a vertex", "Largest vertex valence")
        strip(ax[3], d, ["mesh_volume_ml"], "mL", "Enclosed volume")
        strip(ax[4], d, [c for c in d.columns if c.startswith("cap_") and c.endswith("_area_ratio")], "ratio", "Cap area / opening oval")
        strip(ax[5], d, [c for c in d.columns if c.startswith("cap_") and c.endswith("_plane_rms_mm")], "mm", "Cap flatness (RMS from its plane)")
        pages.append(fig)

    flagged = z.loc[z.abs().max(axis=1) > FLAG_Z] if len(z) else z
    fig, ax = plt.subplots(1, 2, figsize=(13, max(4.5, 0.32 * max(len(flagged), 6) + 1.5)),
                           gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle(f"Cases to review: robust z beyond {FLAG_Z} on at least one index", fontsize=11, color=INK, x=0.01, ha="left")
    if len(flagged):
        im = ax[0].imshow(flagged.values, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
        ax[0].set_xticks(range(flagged.shape[1]), [FLAG_INDICES[c] for c in flagged.columns], rotation=90, fontsize=6)
        ax[0].set_yticks(range(len(flagged)), [c.replace("cas_", "") for c in flagged.index], fontsize=7)
        ax[0].grid(False)
        fig.colorbar(im, ax=ax[0], fraction=0.03, label="robust z")
    else:
        ax[0].text(0.5, 0.5, "no case is beyond the limit", ha="center")
    counts = (z.abs() > FLAG_Z).sum().sort_values()
    counts = counts[counts > 0]
    ax[1].barh([FLAG_INDICES[c] for c in counts.index], counts.values, color=BLUE, height=0.6)
    ax[1].grid(axis="y", visible=False)
    ax[1].set_xlabel("cases beyond the limit")
    ax[1].set_title(f"{len(review)} of {len(d)} cases listed in review_cases.csv")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pages.append(fig)
    return pages

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_dir", type=Path, default=Path(r"C:\data\rh_test\cases"))
    parser.add_argument("--selection", type=Path, required=True, help="selection.txt: one case id per line")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for line in args.selection.read_text().split():
        case_dir = args.cases_dir / f"cas_{int(line):04d}"
        try:
            rows.append(read_case(case_dir))
        except (FileNotFoundError, KeyError) as err:
            missing.append(f"{case_dir.name}: {err}")
    d = pd.DataFrame(rows)
    z, review = flag_cases(d)
    d.to_csv(args.out_dir / "cohort_qc.csv", index=False)
    review.to_csv(args.out_dir / "review_cases.csv", index=False)

    with PdfPages(args.out_dir / "cohort_qc.pdf") as pdf:
        for i, fig in enumerate(build_pages(d, z, review), 1):
            if fig.get_axes() and len(fig.get_axes()) > 2:
                fig.tight_layout(rect=(0, 0, 1, 0.95))
            fig.savefig(args.out_dir / f"cohort_qc_{i}.png", dpi=140)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"{len(d)} cases read, {len(missing)} without complete reports, {len(review)} to review -> {args.out_dir}")
    for m in missing[:10]:
        print("  missing:", m)