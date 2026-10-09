"""
Quality check screening table for all cases that have ts_heart.nii.gz output of screening part.
One rwo per case, with what is needed to assess inclusion prior to RUNNING FULL DATASET INFERENCE:

image     : voxel spacing, z extent, in-plane field of view, direction matrix
  contrast  : median HU per structure, noise (robust SD inside the LV blood pool),
              RA/RV contrast against the wall (delta HU and contrast-to-noise)
  geometry  : volumes, whether RA/RV/PA touch the image border (cut off),
              wall-to-low-density distance on the RA/RV free walls (see below)
  phase     : atrium / ventricle volume ratios (LA/LV, RA/RV): atria are largest at end of
              systole, ventricles at end of diastole, so both ratios move together with
              phase. LV cavity / LV myocardium is kept as a third proxy but is confounded
              by wall thickness (hypertrophy makes it small in any phase)

Wall-to-low-density distance: on the free walls (mask boundary away from other heart
structures) we measure how far the mask boundary is from the nearest voxel below -30 HU
(epicardial fat or lung). That outer edge is visible in EVERY scan, enhanced or not.

In very enhanced cases the endocardial edge is real (not ground truth though), so their values give a
reference for what a correct mask looks like.
Very noisy washout cases with low contrast whose mask sit at a similar distance are geometrically consistent.

Writes <root>/screen_table.csv and two plots (contrast, phase). No inclusion decision is
made here: thresholds are chosen after looking at the distributions.

Caveat: when the domain being segmented touches the boundary of FOV, the report defaults measurement to 0
"""

import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

STRUCT=ndi.generate_binary_structure(3,1)
LOW_DENSITY_HU=-30.0  # fat and lung are below this; blood and muscle are above
FREE_WALL_MARGIN_MM=3.00 # boundary voxels this close to another heart label are not 'free wall'
HEART_NAMES={"myo": "heart_myocardium", "LA": "heart_atrium_left", "LV": "heart_ventricle_left",
               "RA": "heart_atrium_right", "RV": "heart_ventricle_right", "Ao": "aorta",
               "PA": "pulmonary_artery"}

def load(case_dir:Path):
    ct=sitk.ReadImage(str(case_dir/"ct.nii.gz"))
    seg=sitk.ReadImage(str(case_dir/"ts_heart.nii.gz"))
    classes=json.loads((case_dir/"report_heart.json").read_text())["classes"]
    name_to_id={name: int(i) for i, name in classes.items()}
    arr=sitk.GetArrayFromImage(seg)
    masks={short: arr==name_to_id[full] for short, full in HEART_NAMES.items()}
    return ct,sitk.GetArrayFromImage(ct).astype(np.float32),masks

def touches_border(m: np.ndarray) -> bool:
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
                or m[:, :, 0].any() or m[:, :, -1].any())

def robust_sd(values: np.ndarray)->float:
    """SD estimated from the median absolute deviation, not fooled by few outliers"""
    return float(1.4826 * np.median(np.abs(values-np.median(values))))

def free_wall_distance(target: np.ndarray, others:np.ndarray,hu:np.ndarray,spacing)-> float:
    """
    Median distance (mm) from the free-wall boundary of `target` to the nearest
    low-density voxel (fat/lung). 'Free wall' = boundary voxels farther than
    FREE_WALL_MARGIN_MM from any other heart structure.

    """
    margin=[int(np.ceil(15.0 / s)) for s in spacing]
    idx=np.nonzero(target)
    box=tuple(slice(max(int(i.min())-m,0),int(i.max())+m+1) for i,m in zip(idx,margin))
    t,o,h=target[box],others[box],hu[box]
    boundary= t & ~ndi.binary_erosion(t, STRUCT)
    near_other = ndi.distance_transform_edt(~o, sampling=spacing) <= FREE_WALL_MARGIN_MM
    free = boundary & ~near_other
    if free.sum() < 100:
        return float("nan")
    low=(h<LOW_DENSITY_HU) & ~t
    d_low=ndi.distance_transform_edt(~low, sampling=spacing)
    return float(np.median(d_low[free]))

def screen_case(case_dir: Path) -> dict:
    ct, hu, m = load(case_dir)
    spacing = np.array(ct.GetSpacing()[::-1])            # (z, y, x)
    voxel_ml = float(np.prod(spacing)) / 1000.0
    size = ct.GetSize()
    row = {"case": case_dir.name,
           "spacing_xy_mm": round(ct.GetSpacing()[0], 3), "spacing_z_mm": round(ct.GetSpacing()[2], 3),
           "z_extent_mm": round(size[2] * ct.GetSpacing()[2], 1),
           "fov_xy_mm": round(size[0] * ct.GetSpacing()[0], 1),
           "direction": " ".join(f"{d:.0f}" for d in ct.GetDirection())}

    for name, mask in m.items():
        row[f"vol_{name}_ml"] = round(int(mask.sum()) * voxel_ml, 1)
        row[f"hu_{name}"] = round(float(np.median(hu[mask])), 0) if mask.any() else np.nan

    lv_core = ndi.binary_erosion(m["LV"], STRUCT, iterations=3)      # away from the wall
    row["noise_sd_hu"] = round(robust_sd(hu[lv_core]), 1) if lv_core.any() else np.nan
    for ch in ("RA", "RV"):
        delta = row[f"hu_{ch}"] - row["hu_myo"]
        row[f"dhu_{ch}_vs_wall"] = delta
        row[f"cnr_{ch}"] = round(delta / row["noise_sd_hu"], 1) if row["noise_sd_hu"] else np.nan

    for ch in ("RA", "RV", "PA"):
        row[f"cut_{ch}"] = touches_border(m[ch])

    others_all = m["LA"] | m["LV"] | m["myo"] | m["Ao"]
    row["wall_dist_RA_mm"] = round(free_wall_distance(m["RA"], others_all | m["RV"] | m["PA"], hu, spacing), 2)
    row["wall_dist_RV_mm"] = round(free_wall_distance(m["RV"], others_all | m["RA"] | m["PA"], hu, spacing), 2)

    row["phase_la_over_lv"] = round(row["vol_LA_ml"] / row["vol_LV_ml"], 2) if row["vol_LV_ml"] else np.nan
    row["phase_ra_over_rv"] = round(row["vol_RA_ml"] / row["vol_RV_ml"], 2) if row["vol_RV_ml"] else np.nan
    row["phase_lv_over_myo"] = round(row["vol_LV_ml"] / row["vol_myo_ml"], 2) if row["vol_myo_ml"] else np.nan
    return row

def plot_table(rows:list[dict],out_dir:Path)->None:
    names = [r["case"].replace("cas_", "") for r in rows]

    fig, ax = plt.subplots(figsize=(6, 5))
    x = [r["dhu_RA_vs_wall"] for r in rows]
    y = [r["dhu_RV_vs_wall"] for r in rows]
    ax.scatter(x, y)
    for xi, yi, n in zip(x, y, names):
        ax.annotate(n, (xi, yi), fontsize=7)
    ax.set_xlabel("RA median HU - LV wall median HU")
    ax.set_ylabel("RV median HU - LV wall median HU")
    ax.set_title("Right-heart contrast against the wall")
    fig.tight_layout()
    fig.savefig(out_dir / "screen_contrast.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 5))
    x = [r["phase_la_over_lv"] for r in rows]
    y = [r["phase_ra_over_rv"] for r in rows]
    ax.scatter(x, y)
    for xi, yi, n in zip(x, y, names):
        ax.annotate(n, (xi, yi), fontsize=7)
    ax.set_xlabel("LA / LV volume  (higher = more systolic)")
    ax.set_ylabel("RA / RV volume  (higher = more systolic)")
    ax.set_title("Cardiac phase proxies")
    fig.tight_layout()
    fig.savefig(out_dir / "screen_phase.png", dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_dir", type=Path, default=Path(r"C:\data\rh_test\cases"))
    parser.add_argument("--screen_dir", type=Path, required=True, help="has screen_sample.json; table + plots go here")
    args = parser.parse_args()
    out_dir = args.screen_dir

    ids = json.loads((args.screen_dir / "screen_sample.json").read_text())["case_ids"]
    rows = []
    for case_id in ids:
        case_dir = args.cases_dir / f"cas_{case_id:04d}"
        if not (case_dir / "ts_heart.nii.gz").exists():
            print(f"{case_dir.name}: no ts_heart yet, skipped")
            continue
        print(f"{case_dir.name} ...", flush=True)
        rows.append(screen_case(case_dir))

    with open(out_dir/"screen_table.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_table(rows,out_dir)
    print(f"{len(rows)} cases -> {out_dir / 'screen_table.csv'}")

