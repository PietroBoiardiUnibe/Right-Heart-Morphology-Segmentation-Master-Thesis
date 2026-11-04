"""
Step 06d: exploratory data analysis of the SSM input surfaces (06) and the rims (06b) over the cohort, one figure and
one table. Reads per case: surface_chambers_qc.json (06), stl_to_mask.json (04b), rims.json (06b), orifices.json (05b).

Row 1  the surfaces
   A  mesh quality: share of triangles with an angle below 15 deg against edge-length variation (flag > 0.1 %)
   B  volume chain: enclosed volume of the surface (06) and of the mask (04b) against the physicians' STL volume (RA+RV)
   C  footprint area of each vessel opening (the wall patch touching the vessel label)
   D  equivalent diameter of each rim (from the vector area of the K-point loop)
Row 2  rim signatures: distance of rim point j from the rim centroid, seen along the rim's own normal (as 06b samples it),
   against its fixed angle.  Every heart = one thin line; if the correspondence works the lines share one
   shape and differ in size. match = median leave-one-out correlation of each heart with the mean of the others
   (1 = same shape, about 0 = no correspondence).
Row 3
   E  saddle: peak-to-peak distance of the rim from its best plane
   F  distances between opening centres (layout of the right heart)
   G  rim success: openings with a closed star-shaped rim (status ok, one loop, no multiple crossings)
   H  text summary and the cases to look at

Inputs : <cases_dir>/<id>/{surface_chambers_qc.json, stl_to_mask.json, rims.json, orifices.json}
Outputs: <out_dir>/surface_eda.png, surface_eda.csv
Usage  : python 06d_surface_eda.py --cases_dir <cases> --out_dir <qc> [--selection included_cases.txt]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = ("SVC", "IVC", "RVOT", "TV")
COLOURS = {"SVC": "#eb6834", "IVC": "#7a3fb0", "RVOT": "#1baf7a", "TV": "#2a78d6"}
PAIRS = (("SVC", "IVC"), ("SVC", "TV"), ("IVC", "TV"), ("RVOT", "TV"))
MIN_ANGLE_FLAG_PCT = 0.1

def load_case(case):
    qc = json.loads((case / "surface_chambers_qc.json").read_text())
    rep = json.loads((case / "stl_to_mask.json").read_text())["labels"]
    rims = json.loads((case / "rims.json").read_text())
    ori = json.loads((case / "orifices.json").read_text())
    stl = rep["RA"]["stl_ml"] + rep["RV"]["stl_ml"]
    mask = rep["RA"]["mask_ml"] + rep["RV"]["mask_ml"]
    row = {"case": case.name, "vertices": qc["vertices"], "edge_mm": qc["edge_mm_mean"], "edge_cv": qc["edge_cv"],
           "min_angle_deg": qc["min_angle_deg"], "angles_below_pct": qc["angles_below_min_pct"], "max_valence": qc["max_valence"],
           "genus": qc["genus"], "self_intersections": qc["self_intersecting_faces"], "included": qc["included"],
           "problems": " ".join(qc["problems"]),"repaired_vertices":qc.get("repaired_vertices",0), "stl_ml": round(stl, 2), "mask_ml": round(mask, 2), "mesh_ml": qc["volume_ml"],
           "mesh_vs_stl_pct": round(100 * (qc["volume_ml"] / stl - 1), 3), "mask_vs_stl_pct": round(100 * (mask / stl - 1), 3)}
    signature = {}
    for name in NAMES:
        r = rims.get(name, {})
        row[f"{name}_rim_status"] = r.get("status", "missing")
        if name in qc["footprints"]:
            row[f"{name}_fp_area_mm2"] = qc["footprints"][name]["area_mm2"]
            row[f"{name}_fp_loops"] = qc["footprints"][name]["boundary_loops"]
        if r.get("status") == "ok":
            row.update({f"{name}_eq_d_mm": r["eq_diameter_mm"], f"{name}_perimeter_mm": r["perimeter_mm"],
                        f"{name}_saddle_mm": r["nonplanar_pp_mm"], f"{name}_multi": r["multi_crossings"], f"{name}_loops": r["loops_found"]})
            row[f"{name}_axis_vs_rim_deg"] = r["axis_vs_rim_deg"]
            c, n_rim = np.array(r["rim_centroid"]), np.array(r["rim_normal"])
            P = np.array(r["points"]) - c
            P -= np.outer(P @ n_rim, n_rim)                                         # seen along the rim's own normal                                           # seen along the vessel axis
            signature[name] = np.linalg.norm(P, axis=1)
    centres = {n: np.array(ori[n]["centre_world"]) for n in NAMES if "centre_world" in ori.get(n, {})}
    for a, b in PAIRS:
        if a in centres and b in centres:
            row[f"dist_{a}_{b}_mm"] = round(float(np.linalg.norm(centres[a] - centres[b])), 1)
    return row, signature

def loo_corr(M):
    """How well each heart's rim signature matches the average of all the other hearts (1 = same shape, ~0 = unrelated)"""
    return np.array([np.corrcoef(M[i],np.delete(M,i,axis=0).mean(axis=0))[0, 1] for i in range(len(M))])



def strip(ax, groups, labels, colours):
    rng = np.random.default_rng(0)
    for i, (vals, col) in enumerate(zip(groups, colours)):
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals)]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(vals)), vals, s=16, color=col, edgecolor="k", lw=0.3, zorder=3)
        if len(vals):
            ax.hlines(np.median(vals), i - 0.3, i + 0.3, color="k", lw=1.6, zorder=4)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7); ax.tick_params(labelsize=7)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--selection", type=Path, default=None, help="text file, one case id per line (default: every case with rims.json)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [l.strip() for l in args.selection.read_text().split() if l.strip()] if args.selection else None
    rows, sigs, missing = [], {n: {} for n in NAMES}, []
    for case in sorted(p for p in args.cases_dir.iterdir() if p.is_dir()):
        if wanted is not None and case.name not in wanted:
            continue
        if not all((case / f).exists() for f in ("surface_chambers_qc.json", "rims.json", "orifices.json", "stl_to_mask.json")):
            missing.append(case.name); continue
        row, sig = load_case(case)
        rows.append(row)
        for n, v in sig.items():
            sigs[n][case.name] = v
    t = pd.DataFrame(rows)
    t.to_csv(args.out_dir / "surface_eda.csv", index=False)
    print(f"{len(t)} cases read" + (f", {len(missing)} without outputs yet" if missing else ""))

    fig, axes = plt.subplots(3, 4, figsize=(19, 12.5))
    ax = axes[0, 0]                                                                           # A mesh quality
    ax.scatter(t.edge_cv, t.angles_below_pct, s=24, c=np.where(t.included, "#7fbf7f", "#d95f5f"), edgecolor="k", lw=0.3)
    ax.axhline(MIN_ANGLE_FLAG_PCT, color="k", ls="--", lw=0.8)
    ax.set_xlabel("edge-length variation (CV)", fontsize=8); ax.set_ylabel("triangles with an angle < 15 deg (%)", fontsize=8)
    ax.set_title("A  mesh quality (flag above the dashed line)", fontsize=9, loc="left"); ax.tick_params(labelsize=7)
    ax = axes[0, 1]                                                                           # B volume chain
    strip(ax, [t.mask_vs_stl_pct, t.mesh_vs_stl_pct], ["mask vs STL", "surface vs STL"], ["#a9c4e0", "#7fbf7f"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("volume difference (%)", fontsize=8); ax.set_title("B  volume kept: STL -> mask -> surface", fontsize=9, loc="left")
    ax = axes[0, 2]                                                                           # C footprint areas
    strip(ax, [t.get(f"{n}_fp_area_mm2", []) for n in NAMES[:3]], NAMES[:3], [COLOURS[n] for n in NAMES[:3]])
    ax.set_ylabel("footprint area on the wall (mm2)", fontsize=8); ax.set_title("C  footprints (wall touching the vessel)", fontsize=9, loc="left")
    ax = axes[0, 3]                                                                           # D rim diameters
    strip(ax, [t.get(f"{n}_eq_d_mm", []) for n in NAMES], NAMES, [COLOURS[n] for n in NAMES])
    ax.set_ylabel("rim equivalent diameter (mm)", fontsize=8); ax.set_title("D  rim size", fontsize=9, loc="left")

    for j, n in enumerate(NAMES):                                                             # row 2 signatures
        ax = axes[1, j]
        if sigs[n]:
            M = np.array(list(sigs[n].values()))
            theta = 360 * np.arange(M.shape[1]) / M.shape[1]
            for v in M:
                ax.plot(np.append(theta, 360), np.append(v, v[0]), color=COLOURS[n], lw=0.8, alpha=0.5)
            ax.plot(np.append(theta, 360), np.append(M.mean(0), M.mean(0)[0]), color="k", lw=1.8, label="mean")
            ax.legend(fontsize=7, frameon=False)
        ax.set_xticks(range(0, 361, 90)); ax.set_xlabel("fixed angle from the anatomical reference (deg)", fontsize=8)
        ax.set_ylabel("distance of the rim point from the centre (mm)", fontsize=8); ax.tick_params(labelsize=7)
        match = f", match {np.median(loo_corr(np.array(list(sigs[n].values())))):+.2f}" if len(sigs[n]) > 2 else ""
        ax.set_title(f"{n} rim signature ({len(sigs[n])} hearts, K = {len(next(iter(sigs[n].values()))) if sigs[n] else 0}{match})", fontsize=9, loc="left")

    ax = axes[2, 0]                                                                           # E saddle
    strip(ax, [t.get(f"{n}_saddle_mm", []) for n in NAMES], NAMES, [COLOURS[n] for n in NAMES])
    ax.set_ylabel("rim peak-to-peak from its best plane (mm)", fontsize=8); ax.set_title("E  saddle (how non-planar the opening is)", fontsize=9, loc="left")
    ax = axes[2, 1]                                                                           # F layout
    cols = [f"dist_{a}_{b}_mm" for a, b in PAIRS]
    strip(ax, [t.get(c, []) for c in cols], [f"{a}-{b}" for a, b in PAIRS], ["#a3a29c"] * len(PAIRS))
    ax.set_ylabel("distance between opening centres (mm)", fontsize=8); ax.set_title("F  layout of the openings", fontsize=9, loc="left")
    ax = axes[2, 2]                                                                           # G success
    ok = [int(((t[f"{n}_rim_status"] == "ok") & (t.get(f"{n}_loops", 0) == 1) & (t.get(f"{n}_multi", 1) == 0)).sum()) for n in NAMES]
    ax.bar(range(4), ok, color=[COLOURS[n] for n in NAMES], edgecolor="k", lw=0.4)
    ax.axhline(len(t), color="k", ls="--", lw=0.8); ax.set_xticks(range(4)); ax.set_xticklabels(NAMES, fontsize=7); ax.tick_params(labelsize=7)
    ax.set_ylabel("hearts with a closed star-shaped rim", fontsize=8); ax.set_title(f"G  rim success (dashed = all {len(t)})", fontsize=9, loc="left")
    ax = axes[2, 3]                                                                           # H summary
    ax.axis("off")
    bad = t[(~t.included) | (t.angles_below_pct > MIN_ANGLE_FLAG_PCT) | (t.self_intersections > 0)]
    lines = [f"{len(t)} hearts, {int(t.included.sum())} pass the surface QC",
             f"vertices {int(t.vertices.min()):,} to {int(t.vertices.max()):,}, edge {t.edge_mm.min():.2f} to {t.edge_mm.max():.2f} mm",
             f"surface vs STL volume: median {t.mesh_vs_stl_pct.median():+.2f} %, range {t.mesh_vs_stl_pct.min():+.2f} to {t.mesh_vs_stl_pct.max():+.2f} %",
             f"genus {sorted(set(t.genus))}, self-intersections {int(t.self_intersections.sum())}"]
    for n in NAMES:
        if f"{n}_eq_d_mm" in t:
            lines.append(f"{n}: rim d {t[f'{n}_eq_d_mm'].median():.1f} mm ({t[f'{n}_eq_d_mm'].min():.1f}-{t[f'{n}_eq_d_mm'].max():.1f}), "
                         f"saddle {t[f'{n}_saddle_mm'].median():.1f} mm")
    lines.append("repaired surfaces (06): " + (", ".join(t.case[t.repaired_vertices > 0]) or "none"))
    lines.append("look at: " + (", ".join(bad.case) if len(bad) else "none flagged"))
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=8.5, family="monospace")
    ax.set_title("H  summary", fontsize=9, loc="left")
    fig.suptitle("SSM input surfaces (06) and opening rims (06b): exploratory data analysis", fontsize=12)
    fig.tight_layout(); fig.savefig(args.out_dir / "surface_eda.png", dpi=140); plt.close(fig)
    print(f"wrote {args.out_dir / 'surface_eda.png'}")


if __name__ == "__main__":
    main()