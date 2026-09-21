"""
03_mask_to_chamber_meshes.py

The step none of CCT-FM / nnUZoo / CTAug / HolOrama fully hands you: turning a
multi-label NIfTI segmentation mask into clean, watertight, per-chamber surface
meshes suitable as SSM input (ShapeWorks / Deformetrica) and, eventually, CGAL
volumetric meshing.

Pipeline per chamber label:
  1. Isolate the binary mask for that label.
  2. Optionally apply morphological closing to remove small gaps and disconnected
     islands left over from segmentation noise (see the thesis SSM/gating section).
  3. Marching cubes (via VTK) to extract the surface.
  4. Light Laplacian/Taubin smoothing WITHOUT volume shrinkage (Taubin smoothing is
     the standard choice specifically because plain Laplacian smoothing erodes volume,
     which is exactly the failure mode called out in the gating-milestone writeup).
  5. Export as STL (importable into HolOrama for visual QC, or directly into
     ShapeWorks/CGAL).

This is a template: keep_largest_component / smoothing iteration counts are cohort-
and-label-specific and should be tuned once real segmentations are in hand, not
trusted blindly as defaults.
"""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk
from vtk.util import numpy_support


def keep_largest_component(binary_arr: np.ndarray) -> np.ndarray:
    binary_img = sitk.GetImageFromArray(binary_arr.astype(np.uint8))
    cc = sitk.ConnectedComponent(binary_img)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.GetArrayFromImage(relabeled) == 1
    return largest.astype(np.uint8)


def numpy_mask_to_vtk_image(mask: np.ndarray, spacing, origin) -> vtk.vtkImageData:
    vtk_img = vtk.vtkImageData()
    vtk_img.SetDimensions(mask.shape[::-1])  # numpy is (z, y, x); VTK wants (x, y, z)
    vtk_img.SetSpacing(spacing)
    vtk_img.SetOrigin(origin)
    flat = mask.transpose(2, 1, 0).flatten(order="F")
    vtk_arr = numpy_support.numpy_to_vtk(flat.astype(np.uint8), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_img.GetPointData().SetScalars(vtk_arr)
    return vtk_img


def mask_to_smoothed_surface(mask: np.ndarray, spacing, origin, smoothing_iterations: int = 20) -> vtk.vtkPolyData:
    vtk_img = numpy_mask_to_vtk_image(mask, spacing, origin)

    mc = vtk.vtkDiscreteMarchingCubes()
    mc.SetInputData(vtk_img)
    mc.SetValue(0, 1)
    mc.Update()

    # Taubin smoothing: preserves volume far better than plain Laplacian smoothing,
    # which is exactly the failure mode flagged for this step in the thesis writeup.
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(mc.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()

    return normals.GetOutput()


def write_stl(poly_data: vtk.vtkPolyData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(poly_data)
    writer.Write()
    print(f"Wrote {out_path}")


def process_label(mask_nifti_path: Path, label_id: int, out_stl_path: Path, smoothing_iterations: int = 20) -> None:
    seg = sitk.ReadImage(str(mask_nifti_path))
    arr = sitk.GetArrayFromImage(seg)
    binary = (arr == label_id).astype(np.uint8)
    if binary.sum() == 0:
        raise ValueError(f"Label {label_id} not present in {mask_nifti_path}")

    binary = keep_largest_component(binary)
    surface = mask_to_smoothed_surface(binary, seg.GetSpacing(), seg.GetOrigin(), smoothing_iterations)
    write_stl(surface, out_stl_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True, help="Multi-label right-heart NIfTI mask")
    parser.add_argument("--label_id", type=int, required=True, help="e.g. 1=RA, 2=RV per script 02's remap")
    parser.add_argument("--out_stl", type=Path, required=True)
    parser.add_argument("--smoothing_iterations", type=int, default=20)
    args = parser.parse_args()

    process_label(args.mask, args.label_id, args.out_stl, args.smoothing_iterations)
