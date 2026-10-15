"""
Cohort summary of step 06 (uncut chambers + footprints): how many cases pass, why the others fail, and how the
footprints look across the cohort. Read-only.

Usage: python summarize_surfaces.py --cases_dir <cases> --selection <selection.txt> [--out_csv surfaces_06.csv]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

OPENINGS = ("SVC", "IVC", "RVOT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    rows, not_run = [], []
    for case_id in args.selection.read_text().split():
        case = f"cas_{int(case_id):04d}"
        qc_file = args.cases_dir / case / "surface_chambers_qc.json"
        if not qc_file.exists():
            not_run.append(case)
            continue
        qc = json.loads(qc_file.read_text())
        row = {"case": case, "included": qc["included"], "problems": ";".join(qc["problems"]),
               "vertices": qc["vertices"], "volume_ml": qc["volume_ml"], "genus": qc["genus"],
               "self_intersections": qc["self_intersecting_faces"], "min_angle_deg": qc["min_angle_deg"],
               "angles_below_min_pct": qc["angles_below_min_pct"], "tv_vertices": qc["tv_annulus_vertices"]}
        for name in OPENINGS:
            fp = qc["footprints"][name]
            row[f"{name}_faces"] = fp["faces"]
            row[f"{name}_loops"] = fp["boundary_loops"]
            row[f"{name}_opening"] = fp.get("opening")
            row[f"{name}_projected_ratio"] = fp.get("projected_area_ratio", np.nan)
            row[f"{name}_tip_beyond_plane_mm"] = fp.get("tip_beyond_plane_mm", np.nan)
        rows.append(row)

    d = pd.DataFrame(rows)
    print(f"{len(d) + len(not_run)} selected | 06 run on {len(d)} | not run: {not_run or 'none'}")
    print(f"included: {int(d.included.sum())} | excluded: {int((~d.included).sum())}")
    print("problems (a case can have several):")
    for problem, n in Counter(p for ps in d.problems for p in ps.split(";") if p).most_common():
        print(f"   {problem:32s} {n}")
    ok = d[d.included]
    print(f"\nincluded cases ({len(ok)}): vertices {ok.vertices.min()}-{ok.vertices.max()}, "
          f"volume {ok.volume_ml.min():.0f}-{ok.volume_ml.max():.0f} mL, min angle median {ok.min_angle_deg.median():.1f} deg")
    print("footprints of included cases (median [5th-95th percentile]):")
    for name in OPENINGS:
        r, t, f = ok[f"{name}_projected_ratio"], ok[f"{name}_tip_beyond_plane_mm"], ok[f"{name}_faces"]
        print(f"   {name:4s} faces {f.median():.0f} [{f.quantile(.05):.0f}-{f.quantile(.95):.0f}]  "
              f"projected/opening {r.median():.2f} [{r.quantile(.05):.2f}-{r.quantile(.95):.2f}]  "
              f"tip beyond plane {t.median():+.1f} [{t.quantile(.05):+.1f} to {t.quantile(.95):+.1f}] mm")
    print("05 soft flags among included cases (do not affect 06, listed for the record):")
    for name in OPENINGS:
        flagged = ok[ok[f"{name}_opening"] != "ok"]
        if len(flagged):
            print(f"   {name}: {dict(Counter(flagged[f'{name}_opening']))} -> {list(flagged.case)}")
    if args.out_csv:
        d.to_csv(args.out_csv, index=False)
        print(f"\ntable -> {args.out_csv}")
