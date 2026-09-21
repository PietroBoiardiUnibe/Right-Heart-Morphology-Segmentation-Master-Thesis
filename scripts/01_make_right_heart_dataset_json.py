"""
01_make_right_heart_dataset_json.py

Builds an nnU-Net v2 `dataset.json` for your own right-heart cohort, using the exact
label indices CCT-FM/nnUZoo's pretrained Dataset051_CT_Cardio_FULL_Organs model was
trained with (see nnUZoo README / CCT-FM model_training_and_benchmarks/README.md).

Two uses:
  1. Inference only: don't need this at all — nnUNetv2_predict with the pretrained
     checkpoint reproduces all 14 labels regardless of what you ultimately keep.
  2. Fine-tuning on specific annotated right-heart cohort: this script writes a
     dataset.json restricted to the structures actually annotated (by default RA + RV,
     extend as needed), consistent with the source model's label naming so that
     `-pretrained_weights` transfer (matching keys/shapes) still lines up.

Label reference (from nnUZoo, Dataset051_CT_Cardio_FULL_Organs):
  1 Coronary Artery | 2 LV Myocardium | 3 Left Atrium | 4 Left Ventricle | 5 Right Atrium
  6 Right Ventricle | 7 Aorta | 8 Pulmonary Arteries | 9 Precardial Fat | 10 Epicardial Fat
  11 Pulmonary Vein | 12 Superior Vena Cava | 13 Inferior Vena Cava | 14 LA Appendage

NOTE: none of these 14 labels are the tricuspid valve leaflets, chordae or a fine
annular surface. If you fine-tune to add TV-specific structures, add them here as new
label indices (15, 16, ...) after CCT-FM's labels 1-14 (do not renumber the existing
ones if you plan on loading pretrained weights for the shared backbone).
"""

import argparse
import json
from pathlib import Path

RIGHT_HEART_LABELS_DEFAULT = {
    "background": 0,
    "Right Atrium": 5,
    "Right Ventricle": 6,
}


def write_dataset_json(
    out_dir: Path,
    num_training: int,
    labels: dict[str, int] | None = None,
    file_ending: str = ".nii.gz",
) -> Path:
    labels = labels or RIGHT_HEART_LABELS_DEFAULT
    # nnU-Net requires labels to be a contiguous 0..N-1 remap for training on a subset;
    # if you keep the original indices (0, 5, 6) for consistency with the source model's
    # weights, use `--use_nii`/manual remap logic instead — flagged here rather than
    # silently "fixed", since which choice is right depends on whether you are fine-tuning
    # from CCT-FM's checkpoint (keep original indices, remap at data-loading time) or
    # training a fresh right-heart-only model (renumber to 0,1,2 here).
    dataset = {
        "channel_names": {"0": "CT"},
        "labels": labels,
        "numTraining": num_training,
        "file_ending": file_ending,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dataset.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"Wrote {out_path}")
    print(json.dumps(dataset, indent=2))
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="e.g. .../data/nnUNet_raw/Dataset999_RightHeart",
    )
    parser.add_argument("--num_training", type=int, required=True)
    args = parser.parse_args()

    write_dataset_json(args.dataset_dir, args.num_training)
