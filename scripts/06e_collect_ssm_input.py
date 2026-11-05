"""
Step 06e: it is meant to gather the inout fed to the SSM. One run folder holds everything that belongs to one SSM
trial, so the cohort, the random order and the nested subsets are fixed once and never shuffled again.

Mode input (before grooming):   <ssm_dir>/<run>/input/<id>_RH.vtp   copies of the 94 surfaces of 06 (one ShapeWorks domain RH)
                                 <ssm_dir>/<run>/order.txt           the cohort in a seeded random order
                                 <ssm_dir>/<run>/N025/cases.txt ...  nested subsets: the first N cases of order.txt
                                 <ssm_dir>/<run>/input_manifest.csv  per case: size, volume, source checksum, rim status

Mode subsets: (after the grooming step) <ssm_dir>/<run>/N025/groomed/<id>_RH.vtp ...  copies of the ONE grooming, byte-identical
                                 for every N, so that the only thing that changes along the learning curve is the cohort size.

Inputs:  <cases_dir>/<id>/{surface_chambers.vtp, surface_chambers_qc.json, rims.json} (06, 06b), a list of case ids
Usage  : python 06e_collect_ssm_input.py --mode input   --cases_dir <cases> --selection <included_cases.txt> --ssm_dir <ssm> --run 2026-11-05_lc
         python 06e_collect_ssm_input.py --mode subsets --ssm_dir <ssm> --run 2026-11-05_lc
"""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np

CHAMBER="RH"
OPENINGS=("SVC","IVC","RVOT","TV")

def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()

def read_ids(path):
    return [l.strip() for l in path.read_text().split() if l.strip()]

def size_name(n):
    return f"N{n:03d}"


# reproducible shuffle so that the same seeds returns same order on any machine
#nested check is the learning-curve rule: each smaller cohort is a prefix of the following

def collect_input(args):
    run=args.ssm_dir/args.run
    if(run/"order.txt").exists():
        raise SystemExit(f"{run/"order.txt"} exists: the order is part of the trial and is never reshuffled"
                         f"Use a new --run name to start a new trial, please")
    ids=read_ids(args.selection)
    rows=[]
    for cid in ids:
        case=args.cases_dir/cid
        qc=json.loads((case/"surface_chambers_qc.json").read_text())
        rims=json.loads((case/"rims.json").read_text())
        bad=[n for n in OPENINGS if rims[n]["status"] != "ok"]
        if not qc["included"] or bad:
            raise SystemExit(f"{cid} should not enter the SSM as input shape: surface QC included= {qc['included']}, rims not ok: {bad}")
        rows.append({"case": cid, "vertices": qc["vertices"], "faces": qc["faces"], "volume_ml": qc["volume_ml"],
                     "repaired_vertices": qc.get("repaired_vertices", 0), "source_md5": md5(case / "surface_chambers.vtp"),
                     "rims_md5": md5(case / "rims.json")})

    (run/"input").mkdir(parents=True, exist_ok=True)
    for r in rows:
        shutil.copy2(args.cases_dir/r["case"]/"surface_chambers.vtp",run/"input"/f"{r['case']}_{CHAMBER}.vtp")
    order=[str(c) for c in np.random.default_rng(args.seed).permutation(ids)]
    (run / "order.txt").write_text("\n".join(order) + "\n")

    for n in args.sizes:
        if n> len(order):
            raise SystemExit(f"N={n} is larger than the cohort ({len(order)})")
        (run / size_name(n)).mkdir(exist_ok=True)
        (run / size_name(n) / "cases.txt").write_text("\n".join(order[:n]) + "\n")

    with open(run/"input_manifest.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} cases -> {run / 'input'} | seed {args.seed} | subsets {[size_name(n) for n in args.sizes]}")


def make_subsets(args):
    run=args.ssm_dir/args.run
    groomed=run/"groomed"/"shapeworks"
    previous=None
    for n in sorted(args.sizes):
        ids=read_ids(run/size_name(n)/"cases.txt")
        if previous is not None and ids[:len(previous)]!= previous:
            raise SystemExit(f"{size_name(n)} does not contain the smaller subset as its first cases: not nested subset")
        out=run/size_name(n)/"groomed"
        out.mkdir(exist_ok=True)
        for cid in ids:
            src=groomed/f"{cid}_{CHAMBER}.vtp"
            if not src.exists():
                raise SystemExit(f"missing groomed file {src}")
            shutil.copy2(src,out/src.name)
            if md5(src)!=md5(out/src.name):
                raise SystemExit(f"copy of {src.name} differs from the groomed original")
        print(f"{size_name(n)}: {len(ids)} groomed files, identical to the one grooming")
        previous = ids

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode",choices=["input","subsets"],required=True)
    ap.add_argument("--ssm_dir",type=Path, required=True)
    ap.add_argument("--run", required=True,help="run folder name inside ssm_dir, e.g. 2026-11-05_lc")
    ap.add_argument("--cases_dir",type=Path)
    ap.add_argument("--selection",type=Path)
    ap.add_argument("--seed", type=int,default=20261105)
    ap.add_argument("--sizes", type=int, nargs="+",default=[25, 50, 75, 94])
    args=ap.parse_args()
    if args.mode=="input":
        if args.cases_dir is None or args.selection is None:
            raise SystemExit("--mode input needs --cases_dir and --selection")
        collect_input(args)
    else:
        make_subsets(args)

if __name__=="__main__":
    main()

    