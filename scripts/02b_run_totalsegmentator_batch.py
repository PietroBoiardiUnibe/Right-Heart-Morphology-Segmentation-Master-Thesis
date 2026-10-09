"""
02b_run_totalsegmentator_batch.py

Run TotalSegmentator on a batch of ImageCAS cases, in two stages:

  screen : draw a random (seeded) sample of cases from one split-zip group, extract each
           CT into <cases_dir>/cas_XXXX/ct.nii.gz and run ONLY heartchambers_highres.
           The sample is recorded in <screen_dir>/screen_sample.json.
  full   : for the cases listed in a selection file (one case id per line), run the
           vein model (SVC, IVC + pulmonary veins as 'forbidden') and the PA landmarks.

Runs the same on Windows (laptop) and Linux (UBELIX). For a Slurm job array, give
--chunk $SLURM_ARRAY_TASK_ID --n_chunks <array size>: each task then processes every
n_chunks-th case of the list, so tasks never touch the same case.

Every case keeps the fixed folder layout that 04_merge_right_heart_labels.py expects.
Finished outputs are skipped, so the script can be stopped and restarted at any time.
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

# 7-Zip: on the PATH (Linux: conda install p7zip) or the default Windows install folder
SEVEN_ZIP = shutil.which("7z") or shutil.which("7za") or r"C:\Program Files\7-Zip\7z.exe"

# (output key, task, extra TotalSegmentator arguments)
TASKS = {
    "heart": ("heartchambers_highres", []),
    "veins": ("total_v3", ["--roi_subset", "superior_vena_cava", "inferior_vena_cava", "pulmonary_vein",
                           "--higher_order_resampling_LEGACY", "-nr", "4"]),   # smoother upsampling
    "palm": ("pulmonary_artery_landmarks", []),
}
STAGE_TASKS = {"screen": ["heart"], "full": ["veins", "palm"]}


def case_dir_name(case_id: int) -> str:
    return f"cas_{case_id:04d}"


def extract_ct(zip_path: Path, group: str, case_id: int, case_dir: Path, dry_run: bool) -> None:
    """Pull one CT out of the split zip and store it as case_dir/ct.nii.gz."""
    target = case_dir / "ct.nii.gz"
    if target.exists():
        return
    member = f"{group}/{case_id}.img.nii.gz"            # '/' works for 7-Zip on Windows and Linux
    cmd = [SEVEN_ZIP, "e", str(zip_path), f"-o{case_dir}", member, "-y", "-bso0", "-bsp0"]
    print("   ", " ".join(cmd))
    if dry_run:
        return
    case_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    (case_dir / f"{case_id}.img.nii.gz").rename(target)


def run_task(case_dir: Path, key: str, dry_run: bool) -> float:
    """Run one TotalSegmentator task on case_dir/ct.nii.gz; returns runtime in s (0 if skipped)."""
    out = case_dir / f"ts_{key}.nii.gz"
    if out.exists():
        return 0.0
    task, extra = TASKS[key]
    cmd = ["TotalSegmentator", "-i", str(case_dir / "ct.nii.gz"), "-o", str(out), "-ta", task, "--ml",
           "-s", str(case_dir / f"stats_{key}.json"), "-rp", str(case_dir / f"report_{key}.json"), "-q"] + extra
    print("   ", " ".join(cmd))
    if dry_run:
        return 0.0
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def sample_cases(first: int, last: int, n: int, seed: int) -> list[int]:
    """Random sample, so that the selection does not depend on the file order."""
    return sorted(random.Random(seed).sample(range(first, last + 1), n))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["screen", "full"], required=True)
    parser.add_argument("--zip", type=Path, default=Path(r"C:\data\ImageCAS\raw\imagecas\1-200.zip"))
    parser.add_argument("--group", default="1-200", help="folder name inside the zip = case range")
    parser.add_argument("--cases_dir", type=Path, default=Path(r"C:\data\rh_test\cases"))
    parser.add_argument("--screen_dir", type=Path, help="screen: where screen_sample.json is written")
    parser.add_argument("--n", type=int, default=60, help="screen: number of cases to sample (0 = all)")
    parser.add_argument("--seed", type=int, default=42, help="screen: random seed (write it in the thesis)")
    parser.add_argument("--selection", type=Path, help="full: text file with one case id per line")
    parser.add_argument("--chunk", type=int, default=0, help="array task index (0-based)")
    parser.add_argument("--n_chunks", type=int, default=1, help="number of array tasks")
    parser.add_argument("--dry_run", action="store_true", help="only print what would be done")
    args = parser.parse_args()

    if args.stage == "screen":
        if args.screen_dir is None:
            parser.error("--screen_dir is required for --stage screen")
        first, last = (int(v) for v in args.group.split("-"))
        n = args.n or (last - first + 1)
        case_ids = sample_cases(first, last, n, args.seed)
        sample_file = args.screen_dir / "screen_sample.json"
        if args.chunk == 0 and not args.dry_run:         # only one array task writes the record
            args.screen_dir.mkdir(parents=True, exist_ok=True)
            sample_file.write_text(json.dumps({"zip": str(args.zip), "group": args.group, "seed": args.seed,
                                               "case_ids": case_ids}, indent=2))
    else:
        case_ids = [int(line) for line in args.selection.read_text().split() if line.strip()]

    my_ids = case_ids[args.chunk::args.n_chunks]          # this task's share of the list
    print(f"chunk {args.chunk}/{args.n_chunks}: {len(my_ids)} of {len(case_ids)} cases, 7z = {SEVEN_ZIP}")
    t_start = time.time()
    failed = []
    for i, case_id in enumerate(my_ids, 1):
        case_dir = args.cases_dir / case_dir_name(case_id)
        print(f"[{i}/{len(my_ids)}] {case_dir.name}", flush=True)
        try:
            if args.stage == "screen":
                extract_ct(args.zip, args.group, case_id, case_dir, args.dry_run)
            for key in STAGE_TASKS[args.stage]:
                dt = run_task(case_dir, key, args.dry_run)
                if dt:
                    print(f"    {key}: {dt:.0f} s", flush=True)
        except subprocess.CalledProcessError as err:      # one bad case must not stop the batch
            print(f"    FAILED: {err}", flush=True)
            failed.append(case_dir.name)
    print(f"done in {(time.time() - t_start) / 60:.1f} min: {len(my_ids) - len(failed)} ok, {len(failed)} failed {failed}")
    # a few bad cases are tolerated, but if half of the batch fails the cause is systematic
    # (environment, weights, disk): exit with an error so Slurm marks the task FAILED
    if my_ids and len(failed) >= 0.5 * len(my_ids):
        sys.exit(1)


