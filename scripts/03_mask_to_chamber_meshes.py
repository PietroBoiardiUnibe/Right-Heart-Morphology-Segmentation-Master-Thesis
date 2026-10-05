"""
03_mask_to_chamber_meshes.py

The step none of CCT-FM / nnUZoo / CTAug / HolOrama fully hands you: turning a
multi-label NIfTI segmentation mask into clean, watertight, per-chamber surface
meshes suitable as SSM input (ShapeWorks / Deformetrica) and, eventually, CGAL/Gmsh
volumetric meshing.

Pipeline per chamber label:
  1. Isolate the binary mask for that label.
  2. Optionally apply morphological closing to remove small gaps and disconnected
     islands left over from segmentation noise.
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
from vtkmodules.util import numpy_support


def keep_largest_component(binary_arr: np.ndarray) -> np.ndarray:
    binary_img = sitk.GetImageFromArray(binary_arr.astype(np.uint8))
    cc = sitk.ConnectedComponent(binary_img)
    relabeled = sitk.RelabelComponent(cc, sortByObjectSize=True)
    largest = sitk.GetArrayFromImage(relabeled) == 1
    return largest.astype(np.uint8)

def fill_cavities(binary_arr: np.ndarray)-> np.ndarray:
    """
    Function tof ill background cavities which are fully enclosed by the chamber
    (isolated voxels inside blood pool). Marching cubes algorithm would otherwise
    wrap each cavity in its own tiny inner surface, yielding multi-component mesh
    that SSM grooming rejects. Topological adjustment.
    """
    filled=sitk.BinaryFillhole(sitk.GetImageFromArray(binary_arr.astype(np.uint8)))
    return sitk.GetArrayFromImage(filled).astype(np.uint8)


def sitk_to_vtk_image(img: sitk.Image)-> vtk.vtkImageData:
    """
    SimpleITK image to VTK float image, keeping spacing and origin.
    """
    arr=sitk.GetArrayFromImage(img).astype(np.float32)
    vtk_img=vtk.vtkImageData()
    vtk_img.SetDimensions()
    vtk_img.SetSpacing(arr.shape[::-1])
    vtk_img.SetOrigin(img.GetOrigin())
    vtk_img.GetPointData().SetScalars(
        numpy_support.numpy_to_vtk(arr.ravel(), deep=True, array_type=vtk.VTK_FLOAT))
    return vtk_img


def mask_to_smoothed_surface(mask:np.ndarray, spacing,origin, sigma_mm:float=1.0,
                             iso_mm:float=0.8, smoothing_iterations: int=150)-> vtk.vtkPolyData:
    """Binary mask -> smooth closed surface, with all smoothing defined in MILLIMETRES so that scans of
    different voxel size give the same physical smoothing and the same triangle size:
      1. crop to the mask bounding box + margin >= 3 sigma (speed; blur never clips the border)
      2. Gaussian blur of the binary mask, sigma in mm  -> smooth field, no voxel staircases
      3. linear resample onto a common isotropic grid (iso_mm)
      4. marching cubes at 0.5 (iso-level 0.5 of a blurred mask keeps the volume ~constant)
      5. keep the largest surface region (thin 1-2 voxel spikes detach after blurring)
      6. light windowed-sinc smoothing: repairs the sliver triangles marching cubes makes on smooth fields
    """
    img = sitk.GetImageFromArray(mask.astype(np.float32))
    img.SetSpacing(tuple(float(s) for s in spacing))
    img.SetOrigin(tuple(float(o) for o in origin))

    zz,yy,xx=np.nonzero(mask)
    margin = [int(np.ceil(3.0 * sigma_mm / s)) + 2 for s in spacing]
    lo = [max(int(v.min()) - m, 0) for v, m in zip((xx, yy, zz), margin)]
    hi = [min(int(v.max()) + m + 1, n) for v, m, n in zip((xx, yy, zz), margin, img.GetSize())]
    img=sitk.RegionOfInterest(img, [h - l for h, l in zip(hi, lo)], lo) 
    img=sitk.SmoothingRecursiveGaussian(img,sigma_mm)
    size = [int(np.ceil(n * s / iso_mm)) for n, s in zip(img.GetSize(), img.GetSpacing())]
    img = sitk.Resample(img, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(),
                        [iso_mm] * 3, img.GetDirection(), 0.0)  

    mc=vtk.vtkMarchingCubes()
    mc.SetInputData(sitk_to_vtk_image(img))
    mc.SetValue(0,0.5)
    mc.ComputeNormalsOff()
    mc.Update()

    largest=vtk.vtkPolyDataConnectivityFilter()
    largest.SetInputConnection(mc.GetOutputPort())
    largest.SetExtractionModeToLargestRegion()
    clean=vtk.vtkCleanPolyData()
    clean.Update()

    smoother=vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(clean.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)
    smoother.SetPassBand(0.1)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    normals=vtk.vtkPolyDataNormals()
    normals.SetInputConnection(smoother.GetOutputPort())
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()                                # all triangles face OUTWARD
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()



def write_stl(poly_data: vtk.vtkPolyData, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(str(out_path))
    writer.SetFileTypeToBinary()  # ~6x smaller than ASCII, same geometry - to evaluate
    writer.SetInputData(poly_data)
    writer.Write()
    print(f"Wrote {out_path}")


def process_label(mask_nifti_path: Path, label_id: int, out_stl_path: Path, sigma_mm: float = 1.0,
                  iso_mm: float = 0.8, smoothing_iterations: int = 15) -> float:
    seg = sitk.ReadImage(str(mask_nifti_path))
    if not np.allclose(seg.GetDirection(), np.eye(3).ravel()):
        print(f"  WARNING: {mask_nifti_path.name} has a non-identity direction matrix {seg.GetDirection()}; "
              f"mesh coordinates ignore it (axes may be flipped/rotated relative to world space)")
    arr = sitk.GetArrayFromImage(seg)
    binary = (arr == label_id).astype(np.uint8)
    if binary.sum() == 0:
        raise ValueError(f"Label {label_id} not present in {mask_nifti_path}")

    binary = keep_largest_component(binary)
    binary = fill_cavities(binary)                               # removing enclosed cavities
    surface = mask_to_smoothed_surface(binary, seg.GetSpacing(), seg.GetOrigin(), sigma_mm, iso_mm, smoothing_iterations)
    write_stl(surface, out_stl_path)
    return int(binary.sum()) * float(np.prod(seg.GetSpacing())) / 1000.0     # mask volume in mL


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask", type=Path, required=True, help="Multi-label right-heart NIfTI mask")
    parser.add_argument("--label_id", type=int, required=True, help="e.g. 1=RA, 2=RV per script 02's remap")
    parser.add_argument("--out_stl", type=Path, required=True)
    parser.add_argument("--sigma_mm", type=float, default=1.0, help="Gaussian blur of the mask in mm (same value for the whole cohort)")
    parser.add_argument("--iso_mm", type=float, default=0.8, help="isotropic grid spacing in mm before marching cubes")
    parser.add_argument("--smoothing_iterations", type=int, default=15, help="windowed-sinc iterations after marching cubes")
    args = parser.parse_args()

    mask_ml = process_label(args.mask, args.label_id, args.out_stl, args.sigma_mm, args.iso_mm, args.smoothing_iterations)
    print(f"Mask volume {mask_ml:.1f} mL")
