"""
Script to determine which screened cases from dataset go on to the full inference pipeline, with one
written reason per exclusion. Reads the screening table produced earlier and the .json, measures vein coverage
directly on the heart mask for a quality control.

Rules are as follows:
- only cases from the seeded random sample are taken, as hand picking would introduce bias
- complete RA/RV not cut or interrupted by scan FOV border, PA present for landmarks
- outlier RA/RV volume and free-wall distance within median +- 3 robust SD
- veins: scan room above and below the RA shall be >= ROOM_MM so a 1-diameter SVC/IVC stub can fit
necessary, not sufficient: the stub is re-checked on the real vein masks)
- phase: LA/LV <= PHASE_LA_LV_MAX. Only a LEFT-heart proxy, as filtering on RA/RV would cut the very
right-heart shape variation the SSM should learn
- Contrast never excludes a case: the tier (T2: contrast-to-noise ratio CNR>=0.5, T3: below) is recorded 
so the SSM can be rebuilt without a T3 as a sensitivity analysis (to change if a better dataset is found)

Writes <root>/selection.csv (every case, included yes/no, reasons, tier, coverage) and
<root>/selection.txt (one included case id per line, input for 02b --stage full).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk

RULES={
    "pa_min_ml":10.0,
    "robust_z_max":3.0,
    "room_mm":20.0,
    "phase_la_lv_max":0.70,
    "tier2_cnr_min":0.5,
}

def robust_z(x:pd.Series)->pd.Series:
    med=x.median()
    mad=1.4826*(x-med).abs().median()
    return (x-med)/mad

def vein_room(case_dir:Path)-> tuple[float,float]:
    """
    Scan room (mm) above the top and below the bottom of the RA along z. 
    Assumes increasing slice index = superior, as in all cases
    """

    seg=sitk.ReadImage(str(case_dir/"ts_heart.nii.gz"))
    if seg.GetDirection()[8]<=0:
        raise ValueError(f"{case_dir.name}: z axis not pointing up, room above/below would be swapped")
    classes=json.loads((case_dir/"report_heart.json").read_text())["classes"]
    ra_id=next(int(i) for i, name in classes.items() if name == "heart_atrium_right")
    z_slices=np.nonzero((sitk.GetArrayFromImage(seg)==ra_id).any(axis=(1,2)))[0]
    dz=seg.GetSpacing()[2]
    n_z=seg.GetSize()[2]
    return (n_z -1 -z_slices[-1]) * dz, z_slices[0]*dz

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_dir", type=Path, default=Path(r"C:\data\rh_test\cases"))
    parser.add_argument("--screen_dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.screen_dir

    d = pd.read_csv(root / "screen_table.csv")
    sample = {f"cas_{i:04d}" for i in json.loads((root / "screen_sample.json").read_text())["case_ids"]}

    rooms = [vein_room(args.cases_dir / case) for case in d.case]
    d["room_above_RA_mm"]=[r[0] for r in rooms]
    d["room_below_RA_mm"]=[r[1] for r in rooms]
    d["cnr_min"]=d[["cnr_RA","cnr_RV"]].min(axis=1)
    d["tier"]=np.where(d.cnr_min>=RULES["tier2_cnr_min"],"T2","T3")

    checks={
        "not_in_random_sample":~d.case.isin(sample),
        "RA_cut":d.cut_RA,
        "RV_cut":d.cut_RV,
        "PA_missing":d.vol_PA_ml<RULES["pa_min_ml"],
        "volume_outlier":(robust_z(d.vol_RA_ml).abs()> RULES["robust_z_max"]) | (robust_z(d.vol_RV_ml).abs()> RULES["robust_z_max"]),
        "wall_distance_outlier": (robust_z(d.wall_dist_RA_mm).abs() > RULES["robust_z_max"])
                                 | (robust_z(d.wall_dist_RV_mm).abs() > RULES["robust_z_max"]),
        "no_room_for_SVC_stub": d.room_above_RA_mm<RULES["room_mm"],
        "no_room_for_IVC_stub": d.room_below_RA_mm<RULES["room_mm"],
        "phase_outside_band":d.phase_la_over_lv>RULES["phase_la_lv_max"],
    }
    d["reasons"] = [";".join(name for name, failed in checks.items() if failed[i]) for i in range(len(d))]
    d["included"] = d.reasons == ""

    cols = ["case", "included", "reasons", "tier", "cnr_min", "room_above_RA_mm", "room_below_RA_mm",
            "phase_la_over_lv", "phase_ra_over_rv", "vol_RA_ml", "vol_RV_ml", "vol_PA_ml"]
    d[cols].to_csv(root / "selection.csv", index=False)
    ids = [int(c.replace("cas_", "")) for c in d.case[d.included]]
    (root / "selection.txt").write_text("\n".join(str(i) for i in ids) + "\n")
    (root / "selection_rules.json").write_text(json.dumps(RULES, indent=2))

    print("cases failing each check (a case can fail several):")
    for name, failed in checks.items():
        print(f"  {name:24s} {int(failed.sum())}")
    print(f"included: {int(d.included.sum())} / {len(d)}  "
          f"(tier T2: {int((d.included & (d.tier == 'T2')).sum())}, T3: {int((d.included & (d.tier == 'T3')).sum())})")
    