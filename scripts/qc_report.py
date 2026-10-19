"""
Full data-QC report of one dataset, from screening to the surfaces that enter the SSM. One PDF + one PNG per page +
tables, in a dated folder. Reuses the merge / openings pages of cohort_qc.py and adds the funnel, the surface
(step 06) pages, the footprints, the mesh-vs-mask volume check and a gallery of every surface.

Sections
  0  funnel        screened -> selected -> openings measured -> surfaces included (-> SSM), with the reasons
  1  merge (04)    volumes, HU, cleanup, vessel-chamber gaps
  2  openings (05) status per opening, size / shape, layout (fill depth = information only: nothing is filled)
  3  surfaces (06) mesh quality, mesh vs mask volume (Bland-Altman), footprints
  4  gallery       every surface, footprints coloured; excluded / flagged cases first
  5  decision      review heatmap (robust z), review_cases.csv, ssm_candidates.txt
  6  synthetic     (optional) QC verdict of synthetic cohorts + their visual QC pages, read from the output folders of
                   ShapeAnalysis/scripts/synthetic_qc_report.py

Usage: python qc_report.py --cases_dir C:\\data\\rh_test\\cases --selection C:\\data\\rh_test\\cohort\\selection.txt
                           [--selection_csv <pooled selection.csv with reasons>] --out_dir C:\\data\\rh_test\\qc\\report\\<date>_<label>
                           [--synthetic_report <synthetic report folder> [<another> ...]]
"""
import argparse
import json
import subprocess
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy

import cohort_qc as cq          # read_case (04 + 05), strip, page, scatter, flag_cases, colours, flag settings

plt.rcParams["figure.max_open_warning"] = 0     # pages are built first and written at the end: many open figures is expected

OPENINGS = ("SVC", "IVC", "RVOT")
SURFACE_FLAGS = {"surf_projected_ratio_SVC": "SVC footprint / lumen", "surf_projected_ratio_IVC": "IVC footprint / lumen",
                 "surf_projected_ratio_RVOT": "RVOT footprint / lumen", "surf_tip_RVOT_mm": "RVOT tip offset",
                 "surf_tip_IVC_mm": "IVC tip offset", "surf_volume_diff_pct": "mesh vs mask volume"}
cq.FLAG_INDICES.update(SURFACE_FLAGS)
cq.MIN_SCALE.update({"surf_projected_ratio": 0.05, "surf_tip": 1.0, "surf_volume_diff": 0.2})


# ---------------------------------------------------------------- 1. read
def read_surface(case_dir: Path) -> dict:
    """Step 06 (uncut chambers + footprints) QC numbers of one case."""
    path = case_dir / "surface_chambers_qc.json"
    if not path.exists():
        return {"surface_run": False}
    s = json.loads(path.read_text())
    row = {"surface_run": True, "surface_included": s["included"], "surface_problems": ";".join(s["problems"]),
           "surf_vertices": s["vertices"], "surf_edge_mm": s["edge_mm_mean"], "surf_edge_cv": s["edge_cv"],
           "surf_min_angle_deg": s["min_angle_deg"], "surf_angles_below_pct": s["angles_below_min_pct"],
           "surf_max_valence": s["max_valence"], "surf_volume_ml": s["volume_ml"], "surf_genus": s["genus"],
           "surf_self_intersections": s["self_intersecting_faces"], "surf_tv_vertices": s["tv_annulus_vertices"]}
    for name in OPENINGS:
        fp = s["footprints"][name]
        row[f"surf_fp_faces_{name}"] = fp["faces"]
        row[f"surf_projected_ratio_{name}"] = fp.get("projected_area_ratio", np.nan)
        row[f"surf_tip_{name}_mm"] = fp.get("tip_beyond_plane_mm", np.nan)
    return row


def read_all(cases_dir: Path, ids) -> pd.DataFrame:
    rows = []
    for i in ids:
        case_dir = cases_dir / f"cas_{int(i):04d}"
        row = cq.read_case(case_dir)                      # 04 + 05 (its old surface_qc.json part is skipped: archived)
        row.update(read_surface(case_dir))
        rows.append(row)
    d = pd.DataFrame(rows)
    d["mask_RA_RV_ml"] = d.vol_RA_ml + d.vol_RV_ml
    d["surf_volume_diff_pct"] = 100 * (d.surf_volume_ml - d.mask_RA_RV_ml) / d.mask_RA_RV_ml
    d["openings_measured"] = d[[f"status_{n}" for n in OPENINGS]].apply(
        lambda r: all(isinstance(v, str) and ("missing" not in v and "short" not in v and "no_lumen" not in v) for v in r), axis=1)
    return d


# ---------------------------------------------------------------- 2. pages
def funnel_page(d, selection_csv):
    fig, ax = cq.page("Funnel: from the screened scans to the surfaces that can enter the SSM", 1, 2, (12, 4.6))
    steps, reasons = [], {}
    if selection_csv is not None:
        sel = pd.read_csv(selection_csv)
        steps.append(("screened", len(sel)))
        reasons["selection (screening rules)"] = pd.Series(
            [r for rs in sel.reasons.dropna() for r in rs.split(";")]).value_counts()
    steps += [("selected", len(d)), ("3 openings measured (05)", int(d.openings_measured.sum())),
              ("surface included (06)", int(d.surface_included.fillna(False).astype(bool).sum()))]
    names, counts = zip(*steps)
    ax[0].barh(range(len(steps))[::-1], counts, color=cq.BLUE, height=0.6)
    for k, (name, n) in enumerate(steps):
        ax[0].text(n, len(steps) - 1 - k, f"  {n}", va="center", fontsize=8, color=cq.INK)
    ax[0].set_yticks(range(len(steps))[::-1], names)
    ax[0].grid(axis="y", visible=False)
    ax[0].set(xlabel="cases", title="Cases left after each step")
    status = pd.Series([f"05 {n}: {d[f'status_{n}'].iloc[i]}" for n in OPENINGS for i in range(len(d))
                        if isinstance(d[f'status_{n}'].iloc[i], str) and d[f'status_{n}'].iloc[i] != "ok"]).value_counts()
    surf = pd.Series([f"06 {p}" for ps in d.surface_problems.fillna("") for p in ps.split(";") if p]).value_counts()
    lines = []
    for title, series in list(reasons.items()) + [("openings (05, hard + soft)", status), ("surfaces (06)", surf)]:
        lines.append(f"{title}:")
        lines += [f"    {k}: {v}" for k, v in series.items()] or ["    none"]
    ax[1].axis("off")
    ax[1].text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=7, color=cq.INK2)
    ax[1].set_title("Why cases dropped out (a case can have several reasons)", loc="left")
    return fig


def merge_and_openings_pages(d):
    """Pages of cohort_qc.py, with the fill depth labelled as information (the new 06 fills nothing)."""
    figs = []
    fig, ax = cq.page("Merge (04): volumes and intensities", 1, 2, (11, 4.6))
    cq.strip(ax[0], d, [f"vol_{lab}_ml" for lab in cq.LABELS], "mL", "Volumes after merge and cleanup")
    cq.strip(ax[1], d, [f"hu_{lab}" for lab in cq.LABELS], "HU", "Mean HU inside each structure")
    figs.append(fig)
    fig, ax = cq.page("Merge (04): what cleanup changed, and how vessels meet chambers", 2, 2, (11, 8))
    cq.strip(ax[0], d, ["lost_SVC_pct", "lost_IVC_pct"], "% of the vessel", "Vessel voxels given to the chambers")
    cq.strip(ax[1], d, ["islands_removed_vox", "filled_cavity_vox"], "voxels", "Islands removed / holes filled", log=True)
    cq.strip(ax[2], d, [c for c in d.columns if c.startswith("gap_")], "mm", "Vessel-to-chamber gap (0.3-0.5 = touching)")
    cq.strip(ax[3], d, [c for c in d.columns if c.startswith("contact_")], "voxels", "Touching voxels (information)", log=True)
    figs.append(fig)
    fig, ax = cq.page("Openings (05): size and shape (measurements only)", 2, 2, (11, 8))
    cq.strip(ax[0], d, [f"diam_{n}_mm" for n in OPENINGS], "mm", "Equivalent diameter")
    cq.strip(ax[1], d, [f"ratio_{n}" for n in OPENINGS], "long / short axis", "Axis ratio (1 = round)")
    cq.strip(ax[2], d, [f"fill_depth_{n}_mm" for n in OPENINGS], "mm", "Chamber end to opening plane (information only)")
    cq.strip(ax[3], d, ["valve_tilt_deg"], "degrees", "Tilt: valve plane vs PA trunk")
    figs.append(fig)
    fig, ax = cq.page("Openings (05): layout of the three openings", 2, 3, (13, 8))
    cq.strip(ax[0], d, [c for c in d.columns if c.startswith("dist_")], "mm", "Distances between opening centres")
    cq.strip(ax[1], d, [c for c in d.columns if c.startswith("angle_")], "degrees", "Angles between opening axes")
    cq.scatter(ax[2], d, "vol_RA_ml", "diam_SVC_mm", "RA volume vs SVC opening")
    cq.scatter(ax[3], d, "vol_RA_ml", "diam_IVC_mm", "RA volume vs IVC opening")
    cq.scatter(ax[4], d, "vol_RV_ml", "diam_RVOT_mm", "RV volume vs RVOT opening")
    cq.scatter(ax[5], d, "dist_SVC_IVC_mm", "vol_RA_ml", "SVC-IVC distance vs RA volume")
    figs.append(fig)
    return figs


def surface_pages(d):
    s = d[d.surface_run.fillna(False).astype(bool)]
    figs = []
    fig, ax = cq.page("Surfaces (06): mesh quality", 2, 3, (13, 8))
    cq.strip(ax[0], s, ["surf_vertices"], "vertices", "Mesh size")
    cq.strip(ax[1], s, ["surf_edge_mm"], "mm", "Mean edge length (target 1.5)")
    cq.strip(ax[2], s, ["surf_min_angle_deg"], "degrees", "Smallest triangle angle")
    cq.strip(ax[3], s, ["surf_angles_below_pct"], "% of angles", "Angles below 15 deg (limit 0.1 %)")
    cq.strip(ax[4], s, ["surf_max_valence"], "edges at a vertex", "Largest vertex valence")
    cq.strip(ax[5], s, ["surf_tv_vertices"], "vertices", "Tricuspid annulus band")
    figs.append(fig)
    fig, ax = cq.page("Surfaces (06): does the mesh enclose the mask's volume?", 1, 2, (12, 4.6))
    mean = 0.5 * (s.surf_volume_ml + s.mask_RA_RV_ml)
    bias, sd = s.surf_volume_diff_pct.mean(), s.surf_volume_diff_pct.std()
    ax[0].scatter(s.mask_RA_RV_ml, s.surf_volume_ml, s=12, color=cq.BLUE, zorder=3)
    lim = [s.mask_RA_RV_ml.min() * 0.95, s.mask_RA_RV_ml.max() * 1.05]
    ax[0].plot(lim, lim, color=cq.MUTED, lw=1)
    ax[0].set(xlabel="mask RA+RV (mL)", ylabel="mesh (mL)", title="Same volume? (line = identity)")
    for level, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
        ax[1].axhline(level, color=cq.INK2, lw=1, ls=style)
    ax[1].scatter(mean, s.surf_volume_diff_pct, s=12, color=cq.BLUE, zorder=3)
    ax[1].set(xlabel="mean of mesh and mask (mL)", ylabel="(mesh - mask) / mask (%)",
              title=f"Bland-Altman: bias {bias:+.2f} %, 95 % limits {bias - 1.96 * sd:+.2f} to {bias + 1.96 * sd:+.2f} %")
    figs.append(fig)
    fig, ax = cq.page("Surfaces (06): footprints (vessel entry zones on the uncut wall)", 2, 2, (11, 8))
    cq.strip(ax[0], s, [f"surf_fp_faces_{n}" for n in OPENINGS], "faces", "Footprint size")
    cq.strip(ax[1], s, [f"surf_projected_ratio_{n}" for n in OPENINGS], "ratio", "Footprint seen along the vessel / lumen area")
    cq.strip(ax[2], s, [f"surf_tip_{n}_mm" for n in OPENINGS], "mm", "Chamber tip beyond the opening plane (+ = into the vessel)")
    ok = s[["surf_tip_RVOT_mm", "fill_depth_RVOT_mm"]].dropna()
    ax[3].scatter(ok.fill_depth_RVOT_mm, ok.surf_tip_RVOT_mm, s=12, color=cq.BLUE)
    ax[3].set(xlabel="05: chamber end to annulus plane, mean over the lumen (mm)", ylabel="06: RVOT tip beyond the plane (mm)",
              title="Two views of where the RV ends under the pulmonary annulus")
    figs.append(fig)
    return figs


def render_thumbnail(path: Path, size=230):
    reader = vtk.vtkXMLPolyDataReader(); reader.SetFileName(str(path)); reader.Update()
    poly = reader.GetOutput()
    poly.GetCellData().SetActiveScalars("patch")
    lut = vtk.vtkLookupTable(); lut.SetNumberOfTableValues(13); lut.Build()
    for i in range(13):
        lut.SetTableValue(i, .82, .82, .79, 1)
    for i, rgb in ((10, (.9, .35, .35)), (11, (.3, .5, .9)), (12, (.95, .65, .2))):
        lut.SetTableValue(i, *rgb, 1)
    mapper = vtk.vtkPolyDataMapper(); mapper.SetInputData(poly); mapper.SetScalarModeToUseCellData()
    mapper.SetLookupTable(lut); mapper.SetScalarRange(0, 12)
    actor = vtk.vtkActor(); actor.SetMapper(mapper)
    ren = vtk.vtkRenderer(); ren.SetBackground(1, 1, 1); ren.AddActor(actor)
    win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.AddRenderer(ren); win.SetSize(size, size)
    ren.ResetCamera(); cam = ren.GetActiveCamera(); f = np.array(cam.GetFocalPoint())
    cam.SetPosition(*(f + np.array([0.0, cam.GetDistance(), 0.0]))); cam.SetViewUp(0, 0, 1); ren.ResetCamera()
    win.Render()
    grab = vtk.vtkWindowToImageFilter(); grab.SetInput(win); grab.Update()
    img = grab.GetOutput(); w, h, _ = img.GetDimensions()
    return vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1]


def gallery_pages(d, cases_dir, review, per_row=7, rows=6):
    """Every surface, same view, footprints coloured (SVC red, IVC blue, RVOT orange). Excluded and flagged first."""
    s = d[d.surface_run.fillna(False).astype(bool)].copy()
    flagged = set(review.case) if len(review) else set()
    s["order"] = np.where(~s.surface_included.astype(bool), 0, np.where(s.case.isin(flagged), 1, 2))
    s = s.sort_values(["order", "case"])
    figs, per_page = [], per_row * rows
    for start in range(0, len(s), per_page):
        chunk = s.iloc[start:start + per_page]
        fig, ax = plt.subplots(rows, per_row, figsize=(14, 12.5))
        fig.suptitle("Gallery of surfaces (06), anterior view: SVC red, IVC blue, RVOT orange "
                     "| orange title = excluded, dark title = flagged for review", fontsize=10, color=cq.INK, x=0.01, ha="left")
        for a in ax.ravel():
            a.set_axis_off()
        for a, (_, r) in zip(ax.ravel(), chunk.iterrows()):
            a.imshow(render_thumbnail(cases_dir / r.case / "surface_chambers.vtp"))
            colour = cq.ORANGE if r.order == 0 else (cq.INK if r.order == 1 else cq.MUTED)
            a.set_title(r.case.replace("cas_", "") + (" excluded" if r.order == 0 else ""), fontsize=7, color=colour)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        figs.append(fig)
    return figs


def decision_page(d, z, review):
    flagged = z.loc[z.abs().max(axis=1) > cq.FLAG_Z] if len(z) else z
    fig, ax = plt.subplots(1, 2, figsize=(13, max(4.5, 0.3 * max(len(flagged), 6) + 1.5)), gridspec_kw={"width_ratios": [3, 2]})
    fig.suptitle(f"Decision: cases with robust z beyond {cq.FLAG_Z} on at least one index (review, not exclusion)",
                 fontsize=11, color=cq.INK, x=0.01, ha="left")
    if len(flagged):
        im = ax[0].imshow(flagged.values, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
        ax[0].set_xticks(range(flagged.shape[1]), [cq.FLAG_INDICES[c] for c in flagged.columns], rotation=90, fontsize=6)
        ax[0].set_yticks(range(len(flagged)), [c.replace("cas_", "") for c in flagged.index], fontsize=6)
        ax[0].grid(False)
        fig.colorbar(im, ax=ax[0], fraction=0.03, label="robust z")
    counts = (z.abs() > cq.FLAG_Z).sum().sort_values()
    counts = counts[counts > 0]
    ax[1].barh([cq.FLAG_INDICES[c] for c in counts.index], counts.values, color=cq.BLUE, height=0.6)
    ax[1].grid(axis="y", visible=False)
    ax[1].set(xlabel="cases beyond the limit", title=f"{len(review)} of {len(d)} cases in review_cases.csv")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def synthetic_pages(report_dirs) -> list:
    """Section 6: synthetic cohorts. Reads the OUTPUT folders of ShapeAnalysis/scripts/synthetic_qc_report.py
    (synthetic_qc_table.csv + its PNG pages), so the two repositories stay independent. One overview page comparing
    the runs, then every page of every run."""
    runs = []
    for folder in report_dirs:
        table = pd.read_csv(folder / "synthetic_qc_table.csv")
        accepted = table["accepted"].astype(bool)
        problems = pd.Series([p for ps in table.problems.fillna("") for p in str(ps).split(";") if p]).value_counts()
        runs.append({"run": f"{folder.parent.parent.name}/{folder.parent.name}", "hearts": len(table),
                     "accepted": int(accepted.sum()), "rejected_pct": round(100 * (~accepted).mean(), 1),
                     "failure_modes": ", ".join(f"{k} {v}" for k, v in problems.items()) or "none", "folder": folder})
    fig, ax = cq.page("Synthetic cohorts (ShapeAnalysis 04 + 05): QC verdict per run", 1, 1, (12, 1.6 + 0.45 * len(runs)))
    ax[0].axis("off")
    cells = [[r["run"], r["hearts"], r["accepted"], f"{r['rejected_pct']} %", r["failure_modes"]] for r in runs]
    tab = ax[0].table(cellText=cells, colLabels=["run", "hearts", "accepted", "rejected", "failure modes (hearts)"],
                      loc="upper left", cellLoc="left", colWidths=[0.28, 0.07, 0.08, 0.08, 0.49])
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    figs = [fig]
    for r in runs:
        for png in sorted(r["folder"].glob("synthetic_qc_*.png")):
            img = plt.imread(png)
            fig = plt.figure(figsize=(img.shape[1] / 130, img.shape[0] / 130))
            fig.figimage(img)
            fig.text(0.005, 0.995, f"{r['run']}  |  {png.name}", va="top", fontsize=7, color=cq.INK2)
            figs.append(fig)
    return figs


def git_commit(path: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases_dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True, help="selection.txt of the cohort (one case id per line)")
    parser.add_argument("--selection_csv", type=Path, default=None, help="pooled selection.csv with reasons (funnel from screening)")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--no_gallery", action="store_true", help="skip the surface renders (faster)")
    parser.add_argument("--synthetic_report", type=Path, nargs="*", default=[],
                        help="output folder(s) of ShapeAnalysis synthetic_qc_report.py: adds section 6 (synthetic cohorts)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ids = args.selection.read_text().split()
    d = read_all(args.cases_dir, ids)
    z, review = cq.flag_cases(d.rename(columns={"surface_problems": "mesh_problems"}))
    figs = [funnel_page(d, args.selection_csv)] + merge_and_openings_pages(d) + surface_pages(d)
    if not args.no_gallery:
        figs += gallery_pages(d, args.cases_dir, review)
    figs.append(decision_page(d, z, review))
    n_data_pages = len(figs)
    figs += synthetic_pages(args.synthetic_report)
    with PdfPages(args.out_dir / "qc_report.pdf") as pdf:
        for i, fig in enumerate(figs, 1):
            if i <= n_data_pages and len(fig.get_axes()) <= 6:
                fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(args.out_dir / f"qc_report_{i:02d}.png", dpi=130)
            pdf.savefig(fig)
            plt.close(fig)

    d.to_csv(args.out_dir / "cohort_qc.csv", index=False)
    review.to_csv(args.out_dir / "review_cases.csv", index=False)
    ssm = d[d.surface_included.fillna(False).astype(bool)].case.str[4:]
    (args.out_dir / "ssm_candidates.txt").write_text("\n".join(ssm) + "\n")
    run = {"date": str(date.today()), "cases_dir": str(args.cases_dir), "selection": str(args.selection),
           "selection_csv": str(args.selection_csv), "n_cases": len(d), "n_ssm_candidates": len(ssm),
           "synthetic_reports": [str(p) for p in args.synthetic_report],
           "imagesegmentation_commit": git_commit(Path(__file__).resolve().parent)}
    (args.out_dir / "run.json").write_text(json.dumps(run, indent=2))
    print(f"{len(d)} cases, {len(ssm)} SSM candidates, {len(review)} to review, {len(figs)} pages -> {args.out_dir}")
