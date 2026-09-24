"""
02_Inference_Right_Heart.py

Thin wrapper around nnUZoo's 'nnUNetv2_predict_from_modelfolder' to run the pre-trained CCT-FM
14-structure model on the newly introduced CCT volumes and to extract the right heart chambers only
as a separate NIfTI mask comprising labels RA=5 and RV=6 (remapped to 1=RA, 2=RV).

Requires:
- nnUZoo installed, as per environment/ENVIRONMENT.md
- the pretrained checkpoint. ("Model weights are released separately") — place the
    <trainer>__<plans>__3d_fullres/ folder (dataset.json, plans.json, fold_X/checkpoint_*.pth)
    anywhere under data/nnUNet_results/. It is found automatically; --model_dir overrides.
- nnUNet_raw / nnUNet_preprocessed / nnUNet_results env vars: if not set, they default to the
    folders under data/ for this run only.

Single case:   python scripts/02_Inference_Right_Heart.py --case 100206
Whole folder:  python scripts/02_Inference_Right_Heart.py

This script is not responsible for any fine-tuning whatsoever. It is solely for inference with the frozen parameters from the
pretrained model, which is the fastest path to an automated segmentation pipeline.
Fine-tuning on a specific cohort could be considered later, if the annotation effort is worth it.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Fallback ids from Dataset051_CT_Cardio_FULL_Organs; overridden by the model's own dataset.json if it has them.
RIGHT_HEART_LABEL_IDS = {"Right Atrium": 5, "Right Ventricle": 6}


def set_default_nnunet_env() -> None:
    for var, sub in [("nnUNet_raw", "nnUNet_raw"),
                     ("nnUNet_preprocessed", "nnUNet_preprocessed"),
                     ("nnUNet_results", "nnUNet_results")]:
        os.environ.setdefault(var, str(DATA_DIR / sub))


def find_model_dir(results_root: Path, fold: str) -> Path:
    """Locate a <trainer>__<plans>__<config> folder that has fold_<fold>/checkpoint_*.pth."""
    candidates = sorted({ckpt.parent.parent for ckpt in results_root.rglob(f"fold_{fold}/checkpoint_*.pth")})
    candidates = [c for c in candidates if (c / "plans.json").exists() and (c / "dataset.json").exists()]
    if not candidates:
        sys.exit(
            f"No pretrained model found under {results_root}\n"
            f"Expected: <trainer>__<plans>__3d_fullres/{{dataset.json, plans.json, fold_{fold}/checkpoint_final.pth}}\n"
            f"Get the CCT-FM weights from the authors and unpack them there (or pass --model_dir)."
        )
    fullres = [c for c in candidates if c.name.endswith("3d_fullres")]
    if len(fullres) == 1:
        return fullres[0]
    if len(candidates) == 1:
        return candidates[0]
    sys.exit("Several model folders found, pick one with --model_dir:\n  " + "\n  ".join(map(str, candidates)))


def pick_checkpoint(model_dir: Path, fold: str, requested: str | None) -> str:
    fold_dir = model_dir / f"fold_{fold}"
    for name in ([requested] if requested else ["checkpoint_final.pth", "checkpoint_best.pth"]):
        if (fold_dir / name).exists():
            return name
    sys.exit(f"No checkpoint {requested or 'checkpoint_final/best.pth'} in {fold_dir}")


def resolve_label_ids(model_dir: Path) -> dict[str, int]:
    """Take RA/RV ids from the model's dataset.json by name, so a relabelled release can't silently break us."""
    labels = json.loads((model_dir / "dataset.json").read_text())["labels"]
    by_lower = {k.lower().replace("_", " "): v for k, v in labels.items()}
    resolved = {}
    for name, default_id in RIGHT_HEART_LABEL_IDS.items():
        label_id = by_lower.get(name.lower())
        if label_id is None:
            print(f"  WARNING: '{name}' not in model dataset.json labels, falling back to id {default_id}")
            label_id = default_id
        resolved[name] = int(label_id)
    return resolved


def check_input_spacing(model_dir: Path, image_path: Path) -> None:
    plans = json.loads((model_dir / "plans.json").read_text())
    config = model_dir.name.split("__")[-1]
    target = plans.get("configurations", {}).get(config, {}).get("spacing")
    spacing = sitk.ReadImage(str(image_path)).GetSpacing()[::-1]  # plans.json spacing is (z, y, x)
    print(f"  input spacing (z,y,x) = {tuple(round(s, 3) for s in spacing)}   model '{config}' spacing = {target}")
    if target and min(spacing) > min(target) + 1e-3:
        print("  NOTE: input is coarser than the model's working spacing — consider re-running step 00 "
              "with --spacing matching plans.json (or no resampling) to keep fine detail.")


def stage_single_case(input_dir: Path, case: str) -> Path:
    """nnU-Net predicts a whole folder; copy just this case into its own folder."""
    src = input_dir / f"{case}_0000.nii.gz"
    if not src.exists():
        sys.exit(f"{src} not found — run 00_DICOM_to_NIfTI.py first (output must be named <case>_0000.nii.gz)")
    staged = DATA_DIR / "inference_staging" / case
    staged.mkdir(parents=True, exist_ok=True)
    for old in staged.glob("*.nii.gz"):
        old.unlink()
    shutil.copy2(src, staged / src.name)
    return staged


def predictor_executable() -> str:
    # Use the exe next to the running interpreter, so it works without activating the venv (e.g. VS Code debugger).
    local = Path(sys.executable).parent / ("nnUNetv2_predict_from_modelfolder" + (".exe" if os.name == "nt" else ""))
    return str(local) if local.exists() else "nnUNetv2_predict_from_modelfolder"


def run_nnunet_predict(input_dir: Path, output_dir: Path, model_dir: Path, fold: str = "0",
                       checkpoint: str = "checkpoint_final.pth", disable_tta: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # nnUNetv2_predict wants -d/-c/-tr/-p (dataset lookup inside nnUNet_results);
    # pointing directly at a model folder with -m is the _from_modelfolder variant.
    cmd = [
        predictor_executable(),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-m", str(model_dir),
        "-f", fold,
        "-chk", checkpoint,
        # one worker each for pre-/post-processing: avoids Windows multiprocessing + RAM issues on a laptop
        "-npp", "1",
        "-nps", "1",
    ]
    if disable_tta:
        cmd.append("--disable_tta")
    print("Running:", " ".join(f'"{c}"' if " " in c else c for c in cmd))
    subprocess.run(cmd, check=True)


def extract_right_heart_mask(full_mask_path: Path, out_path: Path, label_ids: dict[str, int]) -> None:
    seg = sitk.ReadImage(str(full_mask_path))
    arr = sitk.GetArrayFromImage(seg)
    right_heart = np.zeros_like(arr, dtype=np.uint8)

    for i, (name, label_id) in enumerate(label_ids.items(), start=1):
        right_heart[arr == label_id] = i
        print(f"  label {i} <- '{name}' (source id {label_id}), voxel count = {(arr == label_id).sum()}")
    out_img = sitk.GetImageFromArray(right_heart)
    out_img.CopyInformation(seg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))
    print(f"Wrote right-heart-only mask to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", type=str, default=None, help="Run only this case id (e.g. 100206)")
    parser.add_argument("--input_dir", type=Path, default=DATA_DIR / "inference_input",
                        help="Folder of *_0000.nii.gz CT volumes")
    parser.add_argument("--output_dir", type=Path, default=DATA_DIR / "predictions_full",
                        help="Where full 14-label predictions go")
    parser.add_argument("--right_heart_out_dir", type=Path, default=DATA_DIR / "predictions_right_heart")
    parser.add_argument("--model_dir", type=Path, default=None,
                        help="Pretrained CCT-FM model folder (default: auto-detect under nnUNet_results)")
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint file name inside fold_X/ (default: checkpoint_final.pth, else checkpoint_best.pth)")
    parser.add_argument("--disable_tta", action="store_true", help="Skip mirroring test-time augmentation (~8x faster)")
    args = parser.parse_args()

    set_default_nnunet_env()
    model_dir = args.model_dir or find_model_dir(Path(os.environ["nnUNet_results"]), args.fold)
    checkpoint = pick_checkpoint(model_dir, args.fold, args.checkpoint)
    label_ids = resolve_label_ids(model_dir)
    print(f"Model: {model_dir}\n  fold {args.fold}, {checkpoint}, right-heart ids {label_ids}")

    input_dir = stage_single_case(args.input_dir, args.case) if args.case else args.input_dir
    inputs = sorted(input_dir.glob("*_0000.nii.gz"))
    if not inputs:
        sys.exit(f"No *_0000.nii.gz files in {input_dir}")
    check_input_spacing(model_dir, inputs[0])

    run_nnunet_predict(input_dir, args.output_dir, model_dir, args.fold, checkpoint, args.disable_tta)

    case_ids = {p.name[: -len("_0000.nii.gz")] for p in inputs}
    for full_mask in sorted(args.output_dir.glob("*.nii.gz")):
        if full_mask.name[: -len(".nii.gz")] in case_ids:
            extract_right_heart_mask(full_mask, args.right_heart_out_dir / full_mask.name, label_ids)
