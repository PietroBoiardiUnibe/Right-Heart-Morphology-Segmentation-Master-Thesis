"""
Pool several screen folders (screen table csv and json each) into one cohort.
Select cases script applies the SAME rules to the pooled distribution: robust outlier detection,
outlier limits are computed over all cases together, not per group.

Also reports samples cases that have no row in their screen table, where segmentation failed or
did not run, so missing cases are counted instead of silently disappearing.


Usage: python pool_screens.py <screen_dir> [<screen_dir> ...] --out_dir <cohort_dir>
"""

import argparse
import json
from pathlib import Path

import pandas as pd 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("screens", type=Path, nargs="+")
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()

    tables, case_ids, sources = [], [], []
    for screen in args.screens:
        table = pd.read_csv(screen / "screen_table.csv")
        table["screen"] = screen.name
        sample = json.loads((screen / "screen_sample.json").read_text())
        missing = sorted(set(f"cas_{i:04d}" for i in sample["case_ids"]) - set(table.case))
        print(f"{screen.name}: {len(table)} rows / {len(sample['case_ids'])} sampled, "
              f"missing: {len(missing)} {missing[:10]}{' ...' if len(missing) > 10 else ''}")
        tables.append(table)
        case_ids += sample["case_ids"]
        sources.append({"screen": screen.name, "zip": sample["zip"], "group": sample["group"],
                        "seed": sample["seed"], "n_sampled": len(sample["case_ids"]),
                        "n_screened": len(table), "missing": missing})

    pooled = pd.concat(tables, ignore_index=True)
    duplicates = pooled.case[pooled.case.duplicated()].tolist()
    if duplicates:
        raise ValueError(f"cases screened twice: {duplicates[:10]} -- pool screens of different groups only")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.out_dir / "screen_table.csv", index=False)
    (args.out_dir / "screen_sample.json").write_text(json.dumps(
        {"sources": sources, "case_ids": sorted(case_ids)}, indent=2))
    print(f"pooled {len(pooled)} cases from {len(args.screens)} screens -> {args.out_dir}")