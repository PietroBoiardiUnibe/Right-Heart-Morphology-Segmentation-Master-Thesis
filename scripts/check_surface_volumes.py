"""
Compares for every case in input the volume enclosed by surface_chambers.vtp with the RA+RV voxel volume of mask_rh.
Both are computed independently; neither number comes from a report.

1. Mask vs Mesh with identity line
2. Bland-Altman percentage difference vs mean, with the bias and the 95% limits of agreement
3. difference vs surface/volume ratio, to check if error follows how thin or curved shape is

Sanity check before the SSM gate: does the surface of step 06 enclose the same volume as the mask it was made from?
Mesh volume (divergence theorem on surface_chambers.vtp) vs mask volume (RA + RV voxels of mask_rh x voxel volume).
Smoothing (sigma 1 mm), marching cubes at level 0.5 and re-meshing should change the volume by well under FLAG_PCT;
a larger difference points at a meshing problem (lost piece, flipped triangles, thin part melted by the smoothing).

Output (in --out_dir): surface_volumes.csv, surface_volumes.png
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import SimpleITK as sitk
import vtk
import numpy as np
import pandas as pd
from vtkmodules.util import numpy_support as nps


CHAMBERS=(1,2) # RA, RV labels in mask_rh
FLAG_PCT=2.0 # |mesh - mask| / mask above this (in %) flags the case

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#a3a29c", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 9, "axes.titleweight": "bold",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False})


def mesh_volume_area(path:Path)->tuple[float,float]:
    """Enclosed volume in mL and surface area in mm squared pf a closed triangle mesh"""
    reader=vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly=reader.GetOutput()
    p=nps.vtk_to_numpy(poly.GetPoints().GetData())
    f = nps.vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    a, b, c = p[f[:, 0]], p[f[:, 1]], p[f[:, 2]]
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6000.0        # divergence theorem, mm3 -> mL
    area=0.5 * np.linalg.norm(np.cross(b-a,c-a),axis=1).sum()
    return float(volume),float(area)

def mask_volume(path:Path)->float:
    """RA+RV voxels time the voxel volume (in mL)"""
    img=sitk.ReadImage(str(path))
    voxels=np.isin(sitk.GetArrayViewFromImage(img),CHAMBERS).sum()
    return float(voxels*np.prod(img.GetSpacing())/1000.0)

def label_flagged(ax,d,x,y):
    for _,r in d[d.flagged].iterrows():
        ax.annotate(r.case.replace("cas_",""),(r[x],r[y]),xytext=(3,2),textcoords="offset points",fontsize=6,color=INK2)

def dots(ax, d, x, y):
    ax.scatter(d.loc[~d.flagged, x], d.loc[~d.flagged, y], s=14, color=BLUE, edgecolor=SURFACE, linewidth=0.4, zorder=3)
    ax.scatter(d.loc[d.flagged, x], d.loc[d.flagged, y], s=22, color=ORANGE, edgecolor=SURFACE, linewidth=0.4, zorder=4)
    label_flagged(ax, d, x, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True, help="selection.txt: one case id per line")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for case_id in args.selection.read_text().split():
        case_dir = args.cases_dir / f"cas_{int(case_id):04d}"
        surface=case_dir/"surface_chambers.vtp"
        if not surface.exists():
            continue
        mesh_ml, area=mesh_volume_area(surface)
        rows.append({"case": case_dir.name, "mask_ml": mask_volume(case_dir / "mask_rh.nii.gz"),
                     "mesh_ml": mesh_ml, "area_mm2": area})
    d = pd.DataFrame(rows)
    d["diff_pct"] = 100.0 * (d.mesh_ml - d.mask_ml) / d.mask_ml
    d["mean_ml"] = 0.5 * (d.mesh_ml + d.mask_ml)
    d["surface_per_volume"] = d.area_mm2 / (d.mesh_ml * 1000.0)          # 1/mm: higher = thinner, more curved shape
    d["flagged"] = d.diff_pct.abs() > FLAG_PCT
    d.to_csv(args.out_dir / "surface_volumes.csv", index=False)
    bias,sd=d.diff_pct.mean(), d.diff_pct.std()
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.6))
    fig.suptitle(f"Mesh vs mask volume, {len(d)} cases: bias {bias:+.2f} %, limits of agreement "
                 f"{bias - 1.96 * sd:+.2f} to {bias + 1.96 * sd:+.2f} %, {int(d.flagged.sum())} beyond +-{FLAG_PCT} %",
                 fontsize=11, color=INK, x=0.01, ha="left")

    lo, hi = d[["mask_ml", "mesh_ml"]].min().min() * 0.95, d[["mask_ml", "mesh_ml"]].max().max() * 1.05
    ax[0].plot([lo, hi], [lo, hi], color=MUTED, lw=1, zorder=2)
    dots(ax[0], d, "mask_ml", "mesh_ml")
    ax[0].set(xlabel="mask volume RA+RV (mL)", ylabel="mesh volume (mL)", title="Same volume? (line = identity)")

    for level, style in ((bias, "-"), (bias - 1.96 * sd, "--"), (bias + 1.96 * sd, "--")):
        ax[1].axhline(level, color=INK2, lw=1, ls=style, zorder=2)
    ax[1].axhspan(-FLAG_PCT, FLAG_PCT, color=GRID, alpha=0.5, zorder=1)
    dots(ax[1], d, "mean_ml", "diff_pct")
    ax[1].set(xlabel="mean of mesh and mask (mL)", ylabel="(mesh - mask) / mask (%)",
              title="Bland-Altman: bias (solid), 95 % limits (dashed)")

    dots(ax[2], d, "surface_per_volume", "diff_pct")
    ax[2].set(xlabel="surface / volume (1/mm)", ylabel="(mesh - mask) / mask (%)",
              title="Does the difference follow shape thinness?")

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(args.out_dir / "surface_volumes.png", dpi=140)
    print(f"{len(d)} cases: bias {bias:+.2f} %, SD {sd:.2f} %, flagged {list(d.case[d.flagged]) or 'none'} -> {args.out_dir}")

    









