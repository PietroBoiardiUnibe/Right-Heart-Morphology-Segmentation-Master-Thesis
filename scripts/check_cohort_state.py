"""
Consistency check of the cohort before (re)running a bridge step: for every selected case, are the
04/05 outputs there and in the right order, are 06 outputs left over, what did 04/05 report.
Read-only. Also lists 06 outputs in case folders outside the selection and proposes a clean test case.

Usage: python check_cohort_state.py --cases_dir $CASES --cohort_dir $COHORT
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

NEED = ["mask_rh.nii.gz", "merge_report.json", "mask_rh_cut.nii.gz", "orifices.json"]  # 04 + 05 outputs
STEP06 = ["surface_rh.vtp", "surface_qc.json"]
OPENINGS = ("SVC", "IVC", "RVOT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases_dir", type=Path, required=True)
    parser.add_argument("--cohort_dir", type=Path, required=True)
    args = parser.parse_args()

    ids = [int(x) for x in (args.cohort_dir / "selection.txt").read_text().split()]
    missing, leftover = defaultdict(list), defaultdict(list)
    stale, merge_problems, skipped_by_06, clean = [], [], [], []

    for i in ids:
        d = args.cases_dir / f"cas_{i:04d}"
        for f in NEED:
            if not (d / f).exists():
                missing[f].append(i)
        for f in STEP06:
            if (d / f).exists():
                leftover[f].append(i)
        cut, merged = d / "mask_rh_cut.nii.gz", d / "mask_rh.nii.gz"
        if cut.exists() and merged.exists() and cut.stat().st_mtime < merged.stat().st_mtime:
            stale.append(i)                                  # 05 output older than 04 output
        if (d / "merge_report.json").exists():
            problems = json.loads((d / "merge_report.json").read_text())["problems"]
            if problems:
                merge_problems.append((i, problems))
        if (d / "orifices.json").exists():
            r = json.loads((d / "orifices.json").read_text())
            if any("centre_world" not in r[k] for k in OPENINGS):
                skipped_by_06.append(i)
            elif all(r[k]["status"] == "ok" for k in OPENINGS):
                clean.append(i)

    outside = sorted({p.parent.name for f in STEP06 for p in args.cases_dir.glob(f"cas_*/{f}")
                      if int(p.parent.name[4:]) not in ids})

    print(f"{len(ids)} selected cases")
    print("missing 04/05 outputs:     ", dict(missing) or "none")
    print("05 older than 04 (stale):  ", stale or "none")
    print("leftover 06 outputs:       ", dict(leftover) or "none")
    print("04 merge problems:         ", merge_problems or "none")
    print("06 outputs outside cohort: ", outside or "none")
    print(f"06 will skip (no opening): {len(skipped_by_06)} -> {skipped_by_06}")
    print(f"meshable: {len(ids) - len(skipped_by_06)} | fully clean: {len(clean)} | test case: "
          f"{f'cas_{clean[0]:04d}' if clean else None}")
