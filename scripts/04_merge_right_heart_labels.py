"""
Step 04 of the right-heart segmentation-to-SSM pipeline: the purpose here is to turn the two TotalSegmentator outputs of one case
into ONE clean right-heart label map and measure (not repairing yet) how well the vessels can join the chambers.
Vessel stubs are rebuilt later, from the chamber wall, in 05; this step only aims at delivering clean
chamber masks and trustworthy vessel axes.

heartchambers_highres (sub-mm model) -> RA, RV, PA
total_v3 (1.5 mm model)              -> SVC, IVC

1. merge with priority   chambers win over veins where both claim a voxel
2. clean up              one connected piece per label, enclosed holes filled
3. report & measures         volumes, mean HU, vessel-chamber gap and contact, problems

Case folder (fixed names): ct.nii.gz, ts_heart.nii.gz, ts_veins.nii.gz, report_heart.json,
report_veins.json.   Writes: mask_rh.nii.gz, merge_report.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

RH_LABELS={"RA":1,"RV":2,"PA":3,"SVC":4,"IVC":5}
# (our label, which TotalSegmentator output, class name in that output); the order is the priority

SOURCES=[
    ("RA","heart","heart_atrium_right"),
    ("RV","heart","heart_ventricle_right"),
    ("PA","heart","pulmonary_artery"),
    ("SVC","veins","superior_vena_cava"),
    ("IVC","veins","inferior_vena_cava")
]

# structures that are NOT right-heart blood: cavity filling must never turn them into blood
FORBIDDEN={
    "heart": ("heart_myocardium", "heart_atrium_left", "heart_ventricle_left", "aorta"),
    "veins": ("pulmonary_vein",)
}

# (vessel, chamber it opens into): only used to measure how well they meet
JUNCTIONS=[("SVC","RA"),("IVC","RA"),("PA","RV")]
MAX_GAP_MM=3.0 # a vessel farther than this from its chamber has no usable junction
MIN_CONTACT_VOXELS=20 # fewer touching voxels than this = not connected
STRUCT=ndi.generate_binary_structure(3,1)  # 6-connectivity: voxels touching by a face

# load & merge

def check_same_grid(ref:sitk.Image,img:sitk.Image,name:str)->None:
    """All masks shall sit on the CT grid; otherwise voxel i of one is not voxel i of the other"""
    same=(ref.GetSize()==img.GetSize()
          and np.allclose(ref.GetSpacing(),img.GetSpacing())
          and np.allclose(ref.GetOrigin(),img.GetOrigin())
          and np.allclose(ref.GetDirection(),img.GetDirection())
          )
    if not same:
        raise ValueError(f"{name} is not on the CT grid -- refusing to merge")

#every later step assumes voxel i of the CT is voxel i of each mask. If a mask is on a different grid, the merge would silently mix different places.

def load_case(case_dir:Path):
    """CT image + {output key: (label array, {class name: label id})}. 
     The label numbers are read from TotalSegmentator's own report, never hard-coded.
    """
    ct=sitk.ReadImage(str(case_dir/"ct.nii.gz"))
    maps={}
    for key in ("heart","veins"):
        img=sitk.ReadImage(str(case_dir/ f"ts_{key}.nii.gz"))
        check_same_grid(ct,img, f"ts_{key}")
        classes=json.loads((case_dir/ f"report_{key}.json").read_text())["classes"]
        maps[key]=(sitk.GetArrayFromImage(img),{name: int(i) for i, name in classes.items()})
    
    return ct,maps

def merge_with_priority(maps)->tuple[np.ndarray,dict]:
    """Write the five labels into one array; a voxel already taken is never overwritten"""
    merged=np.zeros(maps["heart"][0].shape,np.uint8)
    report={}

    for ours, key, ts_name in SOURCES:
        arr,name_to_id=maps[key]
        mask=arr==name_to_id[ts_name]
        lost=mask&(merged>0)  # claimed by a higher-priority label
        merged[mask & (merged==0)]= RH_LABELS[ours]
        report[ours] = {"voxels_raw": int(mask.sum()), "lost_to_higher_priority": int(lost.sum())}

    return merged,report

def forbidden_mask(maps)->np.ndarray:
    forbidden=np.zeros(maps["heart"][0].shape,bool)

    for key, names in FORBIDDEN.items():
        arr,name_to_id=maps[key]
        forbidden |= np.isin(arr, [name_to_id[n] for n in names if n in name_to_id])

    return forbidden

# the sub-mm chamber model is more precise than the 1.5 mm vein model, so in the few voxels both claim, the chamber wins. forbidden_mask collects left heart, aorta and pulmonary veins: places a hole-filling step must never turn into blood.

#cleanup

def bounding_box(mask:np.ndarray, margin_vox)->tuple[slice,...]:
    """
    Slices of the box around "mask" grown by margin_vox.
    Distance maps over the whole CT would be slow; the box keeps them cheap.
    """
    idx=np.nonzero(mask)

    return tuple(slice(max(int(i.min())-m,0),min(int(i.max())+m+1,n)) for i,m,n in zip(idx,margin_vox,mask.shape))

def keep_largest_component_per_label(merged: np.ndarray)-> dict:
    """
    Islands like thin spikes or detached blobs shall be eliminated and removed; returns
    the removed voxels per label if applicable.
    """
    removed={}

    for name, label in RH_LABELS.items():
        pieces,n=ndi.label(merged==label,STRUCT)
        if n>1:
            sizes=ndi.sum(np.ones_like(pieces),pieces,index=range(1,n+1))
            drop=(pieces>0) & (pieces != int(np.argmax(sizes)) +1 )
            merged[drop]=0
            removed[name]=int(drop.sum())

    return removed

def fill_cavities(merged: np.ndarray, forbidden:np.ndarray)-> int:
    """
    Holes completely enclosed by the blood pool segmented by algorithm take the label of the nearest
    blood voxel, so the mesh does not get internal surfaces. Holes that are left heart/aorta
    will stay empty.
    """
    box=bounding_box(merged>0,(1,1,1))
    sub,sub_forbidden=merged[box],forbidden[box]
    blood=sub>0
    holes=ndi.binary_fill_holes(blood) & ~blood & ~sub_forbidden
    if holes.any():
        _, nearest=ndi.distance_transform_edt(~blood, return_indices=True)
        sub[holes]=sub[tuple(ix[holes] for ix in nearest)]

    return int(holes.sum())

# a stray island of high intensity (voxel HU) hence becomes a separate surface piece that SSM grooming would reject
# an enclosed hole becomes an internal surface after marching cubes. Both are fixed before meshing

#measuring junctions

def junction_report(merged: np.ndarray, spacing_zyx, vessel: str, chamber: str)->dict:
    """
    How well does the vessel meet the segmented chamber? This is only reported after evaluation, while the
    following script actually rebuilds artificially a stub from the chamber wall to get a standardized truncation artifact.
    In any case, a synthetic outflow or inflow tract extension will be defined for the IB simulation purpose.
    """

    v_mask,c_mask=merged==RH_LABELS[vessel],merged==RH_LABELS[chamber]
    info={"junction": f"{vessel}-{chamber}"}

    if not v_mask.any() or not c_mask.any():
        return {**info, "status": "label_missing"}

    margin=[int(np.ceil(15.0/s)) for s in spacing_zyx]
    box=bounding_box(v_mask | c_mask, margin)
    v,c= v_mask[box], c_mask[box]
    gap=float(ndi.distance_transform_edt(~c, sampling=spacing_zyx)[v].min()) # mm, vessel -> chamber
    contact=int((ndi.binary_dilation(c, STRUCT) & v).sum()) # voxels of the vessel touching it
    info.update(gap_mm=round(gap,2), contact_voxels=contact)
    info["status"]="ok" if (gap<= MAX_GAP_MM and contact>= MIN_CONTACT_VOXELS)
    else "not_attached"
    return info

def write_like(arr:np.ndarray, ref: sitk.Image, path: Path)-> None:
    img=sitk.GetImageFromArray(arr)
    img.CopyInformation(ref) # same spacing, origin and direction as the CT
    sitk.WriteImage(img, str(path), useCompression=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_dir", type=Path, required=True)
    args = parser.parse_args()

    ct, maps = load_case(args.case_dir)
    ct_hu = sitk.GetArrayFromImage(ct).astype(np.float32)
    spacing_zyx = np.array(ct.GetSpacing()[::-1])     # numpy arrays are (z, y, x)

    merged, report = merge_with_priority(maps)
    forbidden = forbidden_mask(maps)
    report["removed_islands_voxels"] = keep_largest_component_per_label(merged)
    report["filled_cavity_voxels"] = fill_cavities(merged, forbidden)
    report["junctions"] = [junction_report(merged, spacing_zyx, v, c) for v, c in JUNCTIONS]

    voxel_ml = float(np.prod(spacing_zyx)) / 1000.0
    for name, label in RH_LABELS.items():
        mask = merged == label
        report[name]["volume_ml"] = round(int(mask.sum()) * voxel_ml, 2)
        report[name]["components"] = int(ndi.label(mask, STRUCT)[1])
        report[name]["mean_hu"] = round(float(ct_hu[mask].mean()), 1) if mask.any() else None
    report["union_components"] = int(ndi.label(merged > 0, STRUCT)[1])

    problems = [f"{j['junction']}:{j['status']}" for j in report["junctions"] if j["status"] != "ok"]
    if report["union_components"] != 1:
        problems.append(f"union_components:{report['union_components']}")
    report["problems"] = problems

    write_like(merged, ct, args.case_dir / "mask_rh.nii.gz")
    (args.case_dir / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(f"{args.case_dir.name}: problems={problems}")
    for j in report["junctions"]:
        print("   ", j)