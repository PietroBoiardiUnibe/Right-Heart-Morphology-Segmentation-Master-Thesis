"""
Step 04b replaces the TotalSegmentator strategy for the physician-validated STL dataset: the label surfaces of one case is turned into
ONE right-heart label map with the same labels as 04 (mask_rh.nii.gz: 1 RA, 2 RV, 3 PA, 4 SVC, 5 IVC),
so that 05b and 06 run on it as before.

1. Read each STL label, merge duplicated vertices and turn the triangles outward (all exported STLs face inward instead),
keep one connected piece: the largest for RA/RV and the one touching its chamber for PA/SVC/IVC (a detached branch
   can be larger than the stub joining the heart); report the removed share and the vessel-chamber gap

2. common grid: chambers' bounding box + MARGIN_MM, isotropic SPACING_MM (default 0.25 mm = the CT slice thickness,
   i.e. no loss against the source voxels of 0.15-0.22 x 0.25 mm); vessels are kept only inside this box

3. Voxelize every label (inside test of closed surface), where labels overlap the priority is RA, RV, PA, SVC, IVC (chambers take over)

4. Report per label: STL volume, mask volume and their relative difference, overlap voxels, pieces removed and so on

Inputs : <stl_dir>/<case>/{RA,RV,Pulmonary Arteries,Superior Vena Cava,Inferior Vena Cava}.stl
Outputs: <out_dir>/<case>/mask_rh.nii.gz, stl_to_mask.json

Usage:  python 04b_stl_to_mask.py --stl_dir <folder of case folders> --out_dir <cases folder> [--ids ...]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import vtk
from vtkmodules.util.numpy_support import vtk_to_numpy, numpy_to_vtk

LABELS={"RA":(1,"RA.stl"), "RV":(2,"RV.stl"), "PA":(3,"Pulmonary Arteries.stl"), "SVC":(4,"Superior Vena Cava.stl"), "IVC":(5,"Inferior Vena Cava.stl")} #order encode priority
JOINS={"PA":"RV","SVC":"RA","IVC":"RA"}
SPACING_MM=0.25
MARGIN_MM=15.0
MAX_REMOVED_PCT=5.0  # more volume than this in discarded pieces: flag
MAX_GAP_MM=3.0  # vessel farther than this from its chamber: no usable junction (flag)

def read_label(path, chamber=None):
    """
    closed surface with outward triangles, reduced to ONE piece:
    chambers -> the largest piece; vessels -> the piece that touches (or comes closest to) its chamber, because a
    detached branch can be larger than the stub that joins the heart. Returns (surface, volume mm3, removed %, gap mm).
    """

    r=vtk.vtkSTLReader(); r.SetFileName(str(path)); r.Update()
    clean=vtk.vtkCleanPolyData(); clean.SetInputData(r.GetOutput());clean.Update()
    whole=clean.GetOutput()
    parts=vtk.vtkPolyDataConnectivityFilter(); parts.SetInputData(whole); parts.SetExtractionModeToAllRegions()
    parts.ColorRegionsOn(); parts.Update()
    region=vtk_to_numpy(parts.GetOutput().GetPointData().GetArray("RegionId"))
    P=vtk_to_numpy(parts.GetOutput().GetPoints().GetData()).astype(float)
    gap=0.0

    if chamber is None:
        keep_id=int(np.argmax(np.bincount(region)))
    else:
        d=distance_to(chamber,P)
        closest=np.array([d[region==k].min() for k in range(region.max() +1 )])
        keep_id, gap=int(np.argmin(closest)),float(closest.min())
    one=vtk.vtkPolyDataConnectivityFilter(); one.SetInputData(whole); one.SetExtractionModeToSpecifiedRegions()
    one.AddSpecifiedRegion(keep_id); one.Update()
    keep=vtk.vtkCleanPolyData(); keep.SetInputData(one.GetOutput()); keep.Update()
    poly=keep.GetOutput()
    P=vtk_to_numpy(poly.GetPoints().GetData()).astype(float)
    F=vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1,3)
    signed=np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]])).sum() / 6

    if signed<0:
        rev=vtk.vtkReverseSense(); rev.SetInputData(poly); rev.Update(); poly= rev.GetOutput()
    total=vtk.vtkMassProperties(); total.SetInputData(whole); total.Update()
    removed=100*(1- abs(signed)/total.GetVolume()) if total.GetVolume() else 0.0

    return poly, abs(signed), max(removed,0.0), gap

def distance_to(surface,X):
    dist=vtk.vtkImplicitPolyDataDistance(); dist.SetInput(surface)
    out=vtk.vtkDoubleArray(); dist.FunctionValue(numpy_to_vtk(np.ascontiguousarray(X),deep=True),out)
    return np.abs(vtk_to_numpy(out))


def voxelise(poly,origin, dims, spacing):
    """1 inside the closed surface, 0 outside, on the given grid (array z,y,x)"""
    img=vtk.vtkImageData(); img.SetSpacing([spacing] *3); img.SetOrigin(*origin); img.SetDimensions(*dims)
    img.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1); img.GetPointData().GetScalars().Fill(1)
    sten=vtk.vtkPolyDataToImageStencil(); sten.SetInputData(poly)
    sten.SetOutputOrigin(*origin); sten.SetOutputSpacing([spacing]*3); sten.SetOutputWholeExtent(img.GetExtent()); sten.Update()
    cut=vtk.vtkImageStencil(); cut.SetInputData(img); cut.SetStencilConnection(sten.GetOutputPort()); cut.SetBackgroundValue(0); cut.Update()
    return vtk_to_numpy(cut.GetOutput().GetPointData().GetScalars()).reshape(dims[::-1]).astype(bool)

def convert(case_dir, out_dir, spacing, margin):
    surfaces,report={},{"spacing_mm":spacing, "margin_mm":margin,"labels":{},"flags":[]}
    for name,(_,fname) in LABELS.items():  # chambers first: vessels are chosen by contact
        surfaces[name],vol,removed, gap= read_label(case_dir/fname, surfaces.get(JOINS.get(name)))
        report["labels"][name]={"stl_ml": round(vol / 1000, 2), "removed_pieces_pct": round(removed, 2)}
        if name in JOINS:
            report["labels"][name]["gap_to_chamber_mm"]=round(gap,2)
            if gap>MAX_GAP_MM:
                report["flags"].append(f"{name}_no_junction")
        if removed > MAX_REMOVED_PCT:
            report["flags"].append(f"{name}_removed_{removed:.0f}pct")
    b=np.array([surfaces[c].GetBounds() for c in ("RA","RV")]).reshape(2,3,2)
    lo,hi=b[:,:,0].min(axis=0)-margin, b[:,:,1].max(axis=0)+ margin
    dims=(np.ceil((hi-lo)/spacing).astype(int) +1 )
    mask=np.zeros(dims[::-1],np.uint8)
    claimed=np.zeros(dims[::-1],np.uint8)

    for name, (label,_) in LABELS.items():
        inside=voxelise(surfaces[name],lo,dims,spacing)
        claimed+= inside
        new=inside & (mask==0)
        mask[new]=label
        mask_ml=inside.sum() * spacing **3 /1000
        entry=report["labels"][name]
        entry.update({"mask_ml": round(float(mask_ml), 2), "kept_ml": round(float(new.sum() * spacing ** 3 / 1000), 2)})

        if name in ("RA","RV"):
            entry["volume_diff_pct"]=round(100*(mask_ml/entry["stl_ml"]-1),2)
    report["overlap_voxels"]=int((claimed>1).sum())
    report["grid"]={"origin": lo.tolist(), "dims_xyz": dims.tolist()}
    img=sitk.GetImageFromArray(mask)
    img.SetSpacing([spacing] *3); img.SetOrigin(lo.tolist())
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img,str(out_dir/"mask_rh.nii.gz"), useCompression=True)
    (out_dir/"stl_to_mask.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stl_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True, help="cases folder; one subfolder per case is written")
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--spacing", type=float, default=SPACING_MM)
    ap.add_argument("--margin", type=float, default=MARGIN_MM)
    args = ap.parse_args()
    for case in sorted(p for p in args.stl_dir.iterdir() if p.is_dir()):
        if args.ids and case.name not in args.ids:
            continue
        rep = convert(case, args.out_dir / case.name, args.spacing, args.margin)
        lab = rep["labels"]
        print(f"{case.name}: RA {lab['RA']['stl_ml']} -> {lab['RA']['mask_ml']} ml ({lab['RA']['volume_diff_pct']:+.2f} %), "
              f"RV {lab['RV']['stl_ml']} -> {lab['RV']['mask_ml']} ml ({lab['RV']['volume_diff_pct']:+.2f} %), "
              f"overlap {rep['overlap_voxels']} vox, flags {' '.join(rep['flags']) or '-'}", flush=True)


if __name__ == "__main__":
    main()