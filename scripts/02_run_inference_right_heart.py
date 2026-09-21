"""
02_run_inference_right_heart.py

Thin wrapper around nnUZoo's `nnUNetv2_predict` to run the pretrained CCT-FM
14-structure model on your own patient CCT volumes and extract just the right-heart
labels (Right Atrium=5, Right Ventricle=6) as a separate NIfTI mask.

Requires:
  - nnUZoo installed (env 1, see environment/ENVIRONMENT.md)
  - the pretrained checkpoint. As of writing, CCT-FM's released weights are linked from
    the CCT-FM repo root README ("Model weights are released separately") — download
    those and point --model_dir at the resulting
    <trainer>__<plans>__3d_fullres/ folder before running this.
  - nnUNet_raw / nnUNet_preprocessed / nnUNet_results env vars set (see ENVIRONMENT.md)

This does NOT fine-tune anything — it is pure inference with the frozen pretrained
model, which is the fastest path to "automated RA/RV segmentation" before it is decided
whether fine-tuning on specific cohort is worth the annotation effort.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RIGHT_HEART_LABEL_IDS = {"Right Atrium": 5, "Right Ventricle": 6}


def run_nnunet_predict(input_dir: Path, output_dir: Path, model_dir: Path, fold: str = "0") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-m", str(model_dir),
        "-f", fold,
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_right_heart_mask(full_mask_path: Path, out_path: Path) -> None:
    seg = sitk.ReadImage(str(full_mask_path))
    arr = sitk.GetArrayFromImage(seg)
    right_heart = np.zeros_like(arr)
    for i, (name, label_id) in enumerate(RIGHT_HEART_LABEL_IDS.items(), start=1):
        right_heart[arr == label_id] = i
        print(f"  label {i} <- '{name}' (source id {label_id}), voxel count = {(arr == label_id).sum()}")
    out_img = sitk.GetImageFromArray(right_heart)
    out_img.CopyInformation(seg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out_img, str(out_path))
    print(f"Wrote right-heart-only mask to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder of *_0000.nii.gz CT volumes")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where full 14-label predictions go")
    parser.add_argument("--model_dir", type=Path, required=True, help="Pretrained CCT-FM model folder")
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--right_heart_out_dir", type=Path, required=True)
    args = parser.parse_args()

    run_nnunet_predict(args.input_dir, args.output_dir, args.model_dir, args.fold)

    for full_mask in sorted(args.output_dir.glob("*.nii.gz")):
        rh_out = args.right_heart_out_dir / full_mask.name
        extract_right_heart_mask(full_mask, rh_out)
