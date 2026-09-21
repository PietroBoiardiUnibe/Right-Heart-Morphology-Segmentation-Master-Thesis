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
