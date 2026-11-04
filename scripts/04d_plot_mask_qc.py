"""
Step 04d (visual QC of 04b): does the label mask reproduce the physicians' STL surfaces?

Per case   cases/<id>/mask_qc.png: three slices (axial, coronal, sagittal) through the centre of the RA|RV junction.
           The mask labels are colour-filled, the physicians' STL surfaces are cut by the same plane and drawn as lines
           on top. Mask edge and STL line must coincide to within a voxel (0.25 mm = a few pixels at this zoom).
Cohort     <out_dir>/mask_qc_summary.png and mask_qc.csv from the 04b reports (cases/<id>/stl_to_mask.json):
           A  STL vs mask volume difference of RA and RV (typically +0.05..+0.1 %, worst ~0.4 % = a 0.06 mm shell)
           B  share of each vessel label discarded by the one-piece rule (vessel islands removed)
           C  gap between each vessel and its chamber (a gap above MAX_GAP_MM = no junction = wrong label)
           D  voxels claimed by two labels (shared boundaries, resolved by priority)

Inputs : <cases_dir>/<id>/mask_rh.nii.gz + stl_to_mask.json (04b), <stl_dir>/<id>/*.stl (raw)
Outputs: cases/<id>/mask_qc.png (per --ids), <out_dir>/mask_qc_summary.png, <out_dir>/mask_qc.csv
Usage  : python 04d_plot_mask_qc.py --cases_dir <cases> --stl_dir <raw STLs> --out_dir <qc> [--ids ...] [--cohort_only]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import SimpleITK as sitk
import vtk
from matplotlib.colors import ListedColormap
from scipy import ndimage as ndi
from vtkmodules.util.numpy_support import vtk_to_numpy

LABELS = {"RA": (1, "RA.stl", "#d03b3b"), "RV": (2, "RV.stl", "#2a78d6"), "PA": (3, "Pulmonary Arteries.stl", "#1baf7a"),
          "SVC": (4, "Superior Vena Cava.stl", "#eb6834"), "IVC": (5, "Inferior Vena Cava.stl", "#7a3fb0")}
CMAP = ListedColormap(["#ffffff00"] + [c for *_, c in LABELS.values()])           # 0 = transparent background
PLANES = {"axial": (2, "x", "y"), "coronal": (1, "x", "z"), "sagittal": (0, "y", "z")}        # normal axis, horizontal, vertical
AXES = {"x": 0, "y": 1, "z": 2}

def read_stl(path):
    r = vtk.vtkSTLReader(); r.SetFileName(str(path)); r.Update()
    c = vtk.vtkCleanPolyData(); c.SetInputData(r.GetOutput()); c.Update()
    return c.GetOutput()


def slice_lines(poly, axis, position):
    """segments (p, q) of the surface cut by the plane coordinate[axis] = position"""
    plane = vtk.vtkPlane(); normal = [0.0, 0.0, 0.0]; normal[axis] = 1.0
    origin = [0.0, 0.0, 0.0]; origin[axis] = position
    plane.SetOrigin(*origin); plane.SetNormal(*normal)
    cut = vtk.vtkCutter(); cut.SetCutFunction(plane); cut.SetInputData(poly); cut.Update()
    out = cut.GetOutput()
    if out.GetNumberOfPoints() == 0:
        return np.empty((0, 2, 3))
    X = vtk_to_numpy(out.GetPoints().GetData())
    L = vtk_to_numpy(out.GetLines().GetConnectivityArray()).reshape(-1, 2)
    return np.stack([X[L[:, 0]], X[L[:, 1]]], axis=1)


def junction_centre(m, spacing, origin):
    """centre (mm) of the RA|RV junction: RA voxels that touch RV voxels"""
    ra, rv = m == 1, m == 2
    touch = ra & ndi.binary_dilation(rv, iterations=2)
    idx = np.argwhere(touch)
    return origin + idx[:, ::-1].mean(axis=0) * spacing

def case_figure(case_dir, stl_dir, out):
    img = sitk.ReadImage(str(case_dir / "mask_rh.nii.gz"))
    m = sitk.GetArrayFromImage(img)                                                   # (z, y, x)
    sp, org = np.array(img.GetSpacing()), np.array(img.GetOrigin())
    c = junction_centre(m, sp, org)
    surfaces = {k: read_stl(stl_dir / case_dir.name / fname) for k, (_, fname, _) in LABELS.items()}
    inside = np.argwhere(m > 0)
    lo, hi = org + inside.min(axis=0)[::-1] * sp - 5, org + inside.max(axis=0)[::-1] * sp + 5
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, (name, (normal, h, v)) in zip(axes, PLANES.items()):
        k = int(round((c[normal] - org[normal]) / sp[normal]))
        sl = (m[k] if normal == 2 else m[:, k, :] if normal == 1 else m[:, :, k]).astype(float)
        hh, vv = AXES[h], AXES[v]
        extent = (org[hh] - sp[hh] / 2, org[hh] + (sl.shape[1] - 0.5) * sp[hh], org[vv] - sp[vv] / 2, org[vv] + (sl.shape[0] - 0.5) * sp[vv])
        ax.imshow(np.ma.masked_equal(sl, 0), cmap=CMAP, vmin=0, vmax=5, origin="lower", extent=extent, interpolation="nearest")
        for key in LABELS:
            for p, q in slice_lines(surfaces[key], normal, c[normal]):
                ax.plot([p[hh], q[hh]], [p[vv], q[vv]], color="#0b0b0b", lw=0.7)
        ax.set_xlim(lo[hh], hi[hh]); ax.set_ylim(lo[vv], hi[vv]); ax.set_aspect("equal")
        ax.set_xlabel(f"{h} (mm)", fontsize=8); ax.set_ylabel(f"{v} (mm)", fontsize=8); ax.tick_params(labelsize=7)
        ax.set_title(f"{name} slice, {'xyz'[normal]} = {c[normal]:.1f} mm", fontsize=9)
    rep = json.loads((case_dir / "stl_to_mask.json").read_text())["labels"]
    handles = [plt.Line2D([0], [0], marker="s", ls="", color=col, label=k) for k, (_, _, col) in LABELS.items()]
    handles.append(plt.Line2D([0], [0], color="#0b0b0b", lw=1, label="physicians' STL"))
    axes[0].legend(handles=handles, fontsize=7, frameon=False, loc="lower left", ncol=2)
    fig.suptitle(f"{case_dir.name}: mask (filled) vs physicians' STL (lines)  |  RA {rep['RA']['stl_ml']} -> {rep['RA']['mask_ml']} ml "
                 f"({rep['RA']['volume_diff_pct']:+.2f} %), RV {rep['RV']['stl_ml']} -> {rep['RV']['mask_ml']} ml ({rep['RV']['volume_diff_pct']:+.2f} %)", fontsize=10)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)

def cohort(cases_dir, out_dir):
    rows = []
    for f in sorted(cases_dir.glob("*/stl_to_mask.json")):
        r = json.loads(f.read_text()); L = r["labels"]
        row = {"case": f.parent.name, "overlap_voxels": r["overlap_voxels"], "flags": " ".join(r["flags"])}
        for k in ("RA", "RV"):
            row[f"{k}_volume_diff_pct"] = L[k]["volume_diff_pct"]
        for k in ("PA", "SVC", "IVC"):
            row[f"{k}_removed_pct"] = L[k]["removed_pieces_pct"]; row[f"{k}_gap_mm"] = L[k]["gap_to_chamber_mm"]
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(out_dir / "mask_qc.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    ax = axes[0]
    for k, col in (("RA", LABELS["RA"][2]), ("RV", LABELS["RV"][2])):
        ax.hist(t[f"{k}_volume_diff_pct"], bins=20, alpha=0.7, color=col, label=f"{k} (max {t[f'{k}_volume_diff_pct'].abs().max():.2f} %)")
    ax.axvline(0.15, color="k", ls="--", lw=0.8); ax.axvline(-0.15, color="k", ls="--", lw=0.8)
    ax.set_xlabel("mask volume vs STL volume (%)", fontsize=8); ax.set_ylabel("cases", fontsize=8); ax.legend(fontsize=7, frameon=False)
    ax.set_title("A  chamber volume preserved", fontsize=9, loc="left")
    ax = axes[1]
    for i, k in enumerate(("PA", "SVC", "IVC")):
        ax.scatter(np.full(len(t), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(t)), t[f"{k}_removed_pct"], s=12, color=LABELS[k][2], edgecolor="k", lw=0.3)
    ax.set_xticks(range(3)); ax.set_xticklabels(["PA", "SVC", "IVC"]); ax.axhline(5, color="k", ls="--", lw=0.8)
    ax.set_ylabel("volume discarded with the extra pieces (%)", fontsize=8); ax.set_title("B  vessel islands removed (flag > 5 %)", fontsize=9, loc="left")
    ax = axes[2]
    for i, k in enumerate(("PA", "SVC", "IVC")):
        ax.scatter(np.full(len(t), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(t)), t[f"{k}_gap_mm"], s=12, color=LABELS[k][2], edgecolor="k", lw=0.3)
    ax.set_xticks(range(3)); ax.set_xticklabels(["PA-RV", "SVC-RA", "IVC-RA"]); ax.axhline(3, color="k", ls="--", lw=0.8)
    for _, r in t[t[["PA_gap_mm", "SVC_gap_mm", "IVC_gap_mm"]].max(axis=1) > 3].iterrows():
        j = int(np.argmax([r.PA_gap_mm, r.SVC_gap_mm, r.IVC_gap_mm])); ax.annotate(r.case, (j, [r.PA_gap_mm, r.SVC_gap_mm, r.IVC_gap_mm][j]), fontsize=6, xytext=(4, 2), textcoords="offset points")
    ax.set_ylabel("distance vessel -> chamber (mm)", fontsize=8); ax.set_title("C  vessel joins its chamber (flag > 3 mm)", fontsize=9, loc="left")
    ax = axes[3]
    ax.hist(t.overlap_voxels, bins=25, color="#a3a29c", edgecolor="k", lw=0.3)
    ax.set_xlabel("voxels claimed by two labels", fontsize=8); ax.set_ylabel("cases", fontsize=8)
    ax.set_title(f"D  shared boundaries (median {int(t.overlap_voxels.median())} vox = {t.overlap_voxels.median() * 0.25 ** 3:.2f} mm3)", fontsize=9, loc="left")
    fig.suptitle(f"STL -> label mask (04b), {len(t)} cases", fontsize=11)
    fig.tight_layout(); fig.savefig(out_dir / "mask_qc_summary.png", dpi=150); plt.close(fig)
    return t

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases_dir", type=Path, required=True)
    ap.add_argument("--stl_dir",type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--ids", nargs="*",help="cases to draw (slices); the cohort figure always uses every case")
    ap.add_argument("--cohort_only", action="store_true")
    args=ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t=cohort(args.cases_dir, args.out_dir)
    print(f"cohort figure: {len(t)} cases, RA max |diff| {t.RA_volume_diff_pct.abs().max():.2f} %, RV {t.RV_volume_diff_pct.abs().max():.2f} %")
    if not args.cohort_only:
        for cid in args.ids or []:
            case_figure(args.cases_dir/cid, args.stl_dir, args.cases_dir/cid/"mask_qc.png")
            print(f"{cid}: mask_qc.png", flush=True)

if __name__=="__main__":
    main()

    
