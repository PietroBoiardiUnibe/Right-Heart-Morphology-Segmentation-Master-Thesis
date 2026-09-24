"""
DICOM to NIfTI conversion script.
Takes a folder of DICOM patient-specific CCT  series and converts it into NIfTI, resampling to a consistent isotropic spacing.
This matches the preprocessing described for the CCT-FM model  by Kazaj et Al. 2026: third order spline interpolation for intensity volume,
nearest-neighbour for any existing label map.

Acquisition specific bits to filled in once real data is available
"""
import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
def resample_to_spacing(image:sitk.Image, target_spacing=(1.0,1.0,1.0), is_label=False)-> sitk.Image:
    original_spacing=image.GetSpacing()
    original_size=image.GetSize()
    new_size=[
        int(round(osz*ospc/tspc))
        for osz,ospc,tspc in zip(original_size, original_spacing, target_spacing)
    ]
    resampler=sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    return resampler.Execute(image)

def convert_series(dicom_dir:Path,out_path:Path,target_spacing=(1.0,1.0,1.0))->None:
    reader=sitk.ImageSeriesReader()
    series_ids=reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    file_names=reader.GetGDCMSeriesFileNames(str(dicom_dir),series_ids[0])
    reader.SetFileNames(file_names)
    image=reader.Execute()

    resampled=resample_to_spacing(image,target_spacing=target_spacing,is_label=False)

    out_path.parent.mkdir(parents=True,exist_ok=True)
    sitk.WriteImage(resampled,str(out_path))
    print(f"Wrote {out_path}  spacing={resampled.GetSpacing()}  size={resampled.GetSize()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dicom_dir", type=Path, required=True, help="Folder containing one DICOM series")
    parser.add_argument("--out", type=Path, required=True, help="Output .nii.gz path")
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    args = parser.parse_args()

    convert_series(args.dicom_dir, args.out, tuple(args.spacing))