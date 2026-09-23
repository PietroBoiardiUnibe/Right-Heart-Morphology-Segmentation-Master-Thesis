"""
02_Inference_Right_Heart.py

Thin wrapper around nnUZoo's 'nnUNetv2_predict' to run the pre-trained CCT-FM 14-structure
model on the newly introduced CCT volumes and to extract the right heart chambers only as a
separate NIfTI mask comprising labels RA=5 and RV=6
 
Requires:
- nnUZoo installed, as per environment/ENVIRONMENT.md
- the pretrained checkpoint. ("Model weights are released separately") — download
    those and point --model_dir at the resulting <trainer>__<plans>__3d_fullres/ folder 
    before running this.
    - nnUnet_raw / nnUne_preprocessed / nnUnet_results env vars set (see ENVIRONMENT.md)

This script is not responsible for any fine-tuning whatsoever. It is solely for inference with the frozen parameters from the 
pretrained model, which is the fastest path to an automated segmentation pipeline"
Fine-tuning on a specific cohort could be considered later, if the annotation effort is worth it.
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk

RIGHT_HEART_LABEL_IDS={"Right Atrium":5,"Right Ventricle":6}

def run_nnunet_predict(input_dir: Path, output_dir: Path, model_dir: Path, fold: str = "0") -> None:
    output_dir.mkdir(parents=True,exist_ok=True)
    cmd=[
        "nnUNetv2_predict",
        "-i",str(input_dir),
        "-o",str(output_dir),
        "-m",str(model_dir),
        "-f",fold
    ]
    print("Running:"," ".join(cmd))
    subprocess.run(cmd,check=True)

def extract_right_heart_mask(full_mask_path: Path, out_path: Path) -> None:
    seg = sitk.ReadImage(str(full_mask_path))
    arr=sitk.GetArrayFromImage(seg)
    right_heart=np.zeros_like(arr)

    for i,(name,label_id) in enumerate(RIGHT_HEART_LABEL_IDS.items(),start=1):
        right_heart[arr==label_id]=i
        print(f"  label {i} <- '{name}' (source id {label_id}), voxel count = {(arr==label_id).sum()}")
    out_img=sitk.GetImageFromArray(right_heart)
    out_img.CopyInformation(seg)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    sitk.WriteImage(out_img,str(out_path))
    print(f"Wrote right-heart-only mask to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder of *_0000.nii.gz CT volumes")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where full 14-label predictions go")
    parser.add_argument("--model_dir", type=Path, required=True, help="Pretrained CCT-FM model folder")
    parser.add_argument("--fold", type=str, default="0")
    parser.add_argument("--right_heart_out_dir", type=Path, required=True)
    args = parser.parse_args()

    run_nnunet_predict(args.input_dir,args.output_dir,args.model_dir,args.fold)

    for full_mask in sorted(args.output_dir.glob("*.nii.gz")):
        rh_out=args.right_heart_out_dir/full_mask.name
        extract_right_heart_mask(full_mask,rh_out)