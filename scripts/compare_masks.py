"""
Compare two multi-label masks of the same case (e.g. laptop vs UBELIX run of the same
TotalSegmentator task): DICE and relative volume difference per label. Used to showcase
that segmentation from different machines could be interchangeable before pooling them altogether.

Usage: python compare_masks.py <mask_a.nii.gz> <mask_b.nii.gz>
"""
import sys

import numpy as np
import SimpleITK as sitk

a_img,b_img=sitk.ReadImage(sys.argv[1]), sitk.ReadImage(sys.argv[2])

if a_img.GetSize() != b_img.GetSize() or not np.allclose(a_img.GetSpacing(),b_img.getSpacing()):
    sys.exit("masks are not on the same grid")

a,b=sitk.GetArrayFromImage(a_img),sitk.GetArrayFromImage(b_img)

for label in sorted((set(np.unique(a)) | set(np.unique(b)))-{0}):
    in_a,in_b =a ==label, b==label

    dice=2*(in_a&in_b).sum()/max(in_a.sum()+in_b.sum(),1)
    vol_diff=100*(int(in_b.sum()) - int(in_a.sum()))/max(int(in_a.sum()),1)
    print(f"label {label:3d}: Dice {dice:.4f}   volume difference {vol_diff:+.2f} %")

