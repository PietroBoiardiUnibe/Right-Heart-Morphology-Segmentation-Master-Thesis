"""
Step 5 of the pipeline for the right-heart mask segmentation. The purpose is to remove the vessels and keep only their openings on the chamber surface,
so that SSM learns only useful anatomy features (the RA, the RV and the RVOT and nothing else).

Input is the merged mask of 04 (mask_rh.nii.gz: 1 RA, 2 RV, 3 PA, 4 SVC, 5 IVC).

  SVC, IVC   opening = flat ellipse at the ostium, measured on the patient's vein:
             1. centreline axis of the segmented vein (centres of two distance shells)
             2. ostium = the vein's contact with the RA, projected onto that axis
             3. elliptical cross-section of the vein, a few mm out of the RA
  RVOT       opening = the pulmonary annulus (valve plane, TotalSegmentator landmark disc),
             with the size of the blood pool in that plane. The pulmonary valve and the PA
             are not modelled

For each opening, inside its ellipse only (the chamber wall around it is never touched):
  - the vessel is deleted,
  - the small recess between the segmented chamber and the opening plane is filled, so the
    opening is exactly flat,
  - a PLUG_MM thick plug (label of the vessel) is left beyond the plane; 06 clips the surface
    exactly on the plane, which removes the plug and leaves a flat elliptical ring.

All info needed afterwards for flow extensions build are written to orifices.json in WORLD coordinates (mm).

Inputs (case folder): mask_rh.nii.gz, ts_heart.nii.gz, ts_veins.nii.gz, ts_palm.nii.gz
and the matching report_*.json.      Outputs: mask_rh_cut.nii.gz, orifices.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

LABELS={"RA":1,"RV":2,"PA":3,"SVC":4,"IVC":5}
PLUG_MM=3.0 #plug kept beyond each opening plane clipped away at the later stage
RECESS_MM=10.0 # recess filled up to the plane from at most this deep inside
FOOTPRINT_MAX_MM=40.0   # nothing beyond this distance from the plane is touched
NEAR_SHELL_MM=(4.0,8.0)   # vein voxels at this distance from the RA: first centreline point
FAR_SHELL_MM=(10.0,15.0) # ... second centreline point (needs >= 15 mm of vein in the scan)
SECTION_WINDOW_MM=(4.0,12.0) # vein cross-section measured here, along the axis from the ostium
RVOT_SLAB_MM=1.5 # RVOT lumen measured in a slab of +- this around the annulus plane
DISC_MARGIN=1.1 # ... and within 1.1 x the radius of the annulus landmark disc
MAX_ELLIPSE_RATIO=3.0  # long / short semi-axis above this: implausible opening, flag
MAX_FILL_DEPTH_MM=6.0   # mean fill depth (fill volume / opening area) above this: flag for review
FORBIDDEN = {"heart": ("heart_myocardium", "heart_atrium_left", "heart_ventricle_left", "aorta"),
             "veins": ("pulmonary_vein",)}
STRUCT=ndi.generate_binary_structure(3,1)

# converting array positions (z,y,x) into world coordinates (x,y,z) applying flipped y-axis once

def to_world_point(img:sitk.Image,p_zyx_mm)->list[float]:
    """
    Array position (z, y, x in mm from voxel 0) -> world point (x, y, z in mm).
    """
    index_xyz=(np.asarray(p_zyx_mm)/np.array(img.GetSpacing()[::-1]))[::-1]
    return list(img.TransformContinuousIndexToPhysicalPoint([float(v) for v in index_xyz]))

def to_world_vector(img:sitk.Image, v_zyx)->list[float]:
    """Direction in array mm space (z, y, x) -> unit world direction (x, y, z); the direction
    matrix is applied, so flipped axes (ImageCAS: y = -1) end up correct."""
    v=np.array(img.GetDirection()).reshape(3,3) @ np.asarray(v_zyx)[::-1]
    return list(v/np.linalg.norm(v))

def load_labels(case_dir:Path, key: str)-> tuple[np.ndarray,dict]:
    arr=sitk.GetArrayFromImage(sitk.ReadImage(str(case_dir/f"ts_{key}.nii.gz")))
    classes=json.loads((case_dir/f"report_{key}.json").read_text())["classes"]
    return arr, {name: int(i) for i,name in classes.items()}

def forbidden_mask(case_dir:Path)->np.ndarray:
    """
    Left heart, aorta, myocardium, pulmonary veins: filling must never reach them.
    """
    forbidden=None
    for key,names in FORBIDDEN.items():
        arr,name_to_id=load_labels(case_dir,key)
        f=np.isin(arr,[name_to_id[n] for n in names if n in name_to_id])
        forbidden=f if forbidden is None else forbidden | f
    return forbidden

# geometry helpers to fit ellipse and measuring opening, make the opening flat, hold the review flags

def along_and_across(points,origin, axis):
    """
    Split points into the coordinate along axis (t,mm) and the vector across it
    """
    rel=points - origin
    t= rel @ axis
    return t, rel - np.outer(t,axis)

def fit_ellipse(section: np.ndarray, area_mm2: float):
    """
    Ellipse of a cross-section (points across axis). For a filled ellipse the SD along
    a principal direction is half the semi-axis; the axes are rescaled so the ellipse has the
    measured area. Return centre offset, semi-axes A>= B, directions.
    """
    centre=section.mean(axis=0)
    _,sv,vt=np.linalg.svd(section -centre, full_matrices=False)
    sd=sv[:2]/np.sqrt(len(section))
    scale=np.sqrt(area_mm2/(np.pi*4.0*sd[0]*sd[1]))
    return centre, 2.0 * sd[0] * scale, 2.0 * sd[1] * scale, vt[0], vt[1]

def footprint_grid(shape,spacing, centre,normal,e1,e2,A,B,depth_mm):
    """
    box around the opening and for every voxel in it: distance t from the plane and whether it lies inside the ellipse footprint
    """
    reach=max(A,depth_mm,FOOTPRINT_MAX_MM)
    lo=np.maximum(np.floor((centre-reach)/spacing).astype(int),0)
    hi=np.minimum(np.ceil((centre+reach)/spacing).astype(int)+1,shape)
    box=tuple(slice(a,b) for a,b in zip(lo,hi))
    grid = np.stack(np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), axis=-1) * spacing
    rel=grid-centre
    t=rel @ normal
    pad=float(spacing.min())
    inside= ((rel @ e1) / (A + pad)) ** 2 + ((rel @ e2) / (B + pad)) ** 2 <= 1.0
    return box, t, inside

def shape_opening(m, forbidden, spacing, box, t, inside, chamber_label, plug_label, removed)->float:
    """
    Inside the ellipse footprint only: fill the recess up to the plane with the chamber label, put a plug of PLUG_MM beyond it
    and clear everything farther out. Returns filled mm3
    """
    sub, sub_forbidden,sub_removed=m[box],forbidden[box],removed[box]
    free=(sub==0) & ~sub_forbidden # nothing there (vessel already deleted)
    recess = inside & (t <= 0) & (t >= -RECESS_MM) & free & (sub_removed | (t >= -2.0))
    sub[recess] = chamber_label
    beyond = inside & (t > 0) & (t <= FOOTPRINT_MAX_MM) & np.isin(sub, (LABELS["RA"], LABELS["RV"], 0))
    sub[beyond]=0  # chamber poking into the vessel lumen
    plug= inside & (t>0) & (t<= PLUG_MM) & (sub==0) & ~sub_forbidden
    sub[plug]=plug_label
    return float(recess.sum())*float(np.prod(spacing))


def opening_quality(filled_mm3, A,B)-> tuple[dict,list]:
    """
    Mean depth of the filled recess (mm) = filled volume / opening area which is
    size-independent, so a wide RVOT and a narrow SVC are judged by the same rule
    """
    depth=filled_mm3/(np.pi*A*B)
    flags=[]
    if A/B > MAX_ELLIPSE_RATIO:
        flags.append("implausible_ellipse")
    if depth> MAX_FILL_DEPTH_MM:
        flags.append("deep_recess_fill")
    return {"recess_filled_ml": round(filled_mm3 / 1000.0, 3), "recess_fill_depth_mm": round(float(depth), 2)}, flags


def opening_record(img,centre,normal,e1,e2,A,B)-> dict:
    """ What the following script will clip and where flow extensions will be needed"""
    return {"centre_world": to_world_point(img,centre),"normal_world": to_world_vector(img,normal),
    "axis1_world":to_world_vector(img,e1),"axis2_world":to_world_vector(img,e2),
    "semi_axes_mm": [round(float(A), 2), round(float(B), 2)],
    "equivalent_diameter_mm": round(float(2.0 * np.sqrt(A * B)), 2)
    }

# SVC and IVC opening
def vein_opening(m, forbidden, spacing, img, name)-> dict:
    """ deleting the vein and leaving a flat elliptical opening at its ostium on the RA"""

    V,RA= m == LABELS[name], m==LABELS["RA"]
    if V.sum()<100:
        return {"status":"vein_missing"}
    voxel_mm3=float(np.prod(spacing))

    #1 centerline axis: centres of 2 shells at fixed distance from RA (shells cut the veins
    #cross-wise so their centres lie on its centerlines)
    d_ra=ndi.distance_transform_edt(~RA, sampling=spacing)
    pts,d_pts=np.argwhere(V)*spacing, d_ra[V]
    near=pts[(d_pts>=NEAR_SHELL_MM[0]) & (d_pts<=NEAR_SHELL_MM[1])]
    far=pts[(d_pts>=FAR_SHELL_MM[0]) & (d_pts<=FAR_SHELL_MM[1])]
    touch_mm= 1.0 + max(0.0, float(d_pts.min())-0.5)  # vessels that stop short of the RA: measure from their closest end
    touching=pts[d_pts <= touch_mm]

    if len(near)<50 or len(far)<50 or len(touching)<5:
        return {"status":"vein_too_short_for_axis"}
    c_near=near.mean(axis=0)
    axis=far.mean(axis=0) - c_near
    axis/= np.linalg.norm(axis) # points out of the RA, into the vein
    #ostium contact with the RA projected onto centerline
    ostium=c_near+ ((touching.mean(axis=0) - c_near) @ axis) * axis
    t, across=along_and_across(pts,ostium,axis)
    sel=(t>=SECTION_WINDOW_MM[0]) & (t<SECTION_WINDOW_MM[1]) & (np.linalg.norm(across,axis=1)<=30.0)
    if sel.sum() < 50:
        return {"status": "vein_too_short_for_section"}
    area=sel.sum() * voxel_mm3 / (SECTION_WINDOW_MM[1] - SECTION_WINDOW_MM[0])
    offset,A,B,e1,e2=fit_ellipse(across[sel],area)
    centre= ostium + offset

    #deleting the vein
    m[V]=0
    box,t_grid,inside=footprint_grid(m.shape, spacing, centre, axis,e1,e2,A,B, RECESS_MM)
    filled=shape_opening(m,forbidden,spacing,box,t_grid,inside,LABELS["RA"],LABELS[name],V)

    quality, flags=opening_quality(filled, A, B)
    info={**opening_record(img,centre,axis,e1,e2,A,B),**quality}
    info["status"]=";".join(flags) if flags else "ok"
    return info

#RVOT opening at the pulmonary annulus: the plane is derived from inferred annulus disc segmentation position
# its thinnest direction is the valve plane normal, the size comes from the blood pool in that plane

def rvot_opening(m,forbidden, spacing,img,case_dir)->dict:
    """
    Delete the PA; leave a flat elliptical outlet in the plane of the pulmonary annulus
    """

    palm,name_to_id=load_labels(case_dir,"palm")
    disc=np.argwhere(palm==name_to_id["pul_annulus"])*spacing
    bif=np.argwhere(palm==name_to_id["pul_bifurcation"])*spacing

    if len(disc)<30 or len(bif)<30:
        return {"status":"landmark_missing"}

    disc_centre=disc.mean(axis=0)
    normal=np.linalg.svd(disc - disc_centre, full_matrices=False)[2][2]#thinnest direction of the disc
    trunk=bif.mean(axis=0)-disc_centre

    if normal @ trunk <0:  # points out of the RV, towards the PA
        normal= -normal

    # lumen in the annulus plane: blood pool (RV + PA) within a thin slab around the plane,
    # limited to the annulus itself (the slab must not pick up RV body lying near the plane)
    _,disc_across=along_and_across(disc,disc_centre,normal)
    annulus_radius=DISC_MARGIN* float(np.percentile(np.linalg.norm(disc_across,axis=1),95))
    PA=m==LABELS["PA"]
    pool=np.argwhere((m==LABELS["RV"]) | PA) * spacing
    t,across=along_and_across(pool,disc_centre,normal)
    sel=(np.abs(t)<=RVOT_SLAB_MM) & (np.linalg.norm(across,axis=1)<= annulus_radius)
    if sel.sum() < 50:
        return {"status": "no_lumen_at_annulus"}

    area=sel.sum() * float(np.prod(spacing))/(2.0 * RVOT_SLAB_MM)
    offset, A,B,e1,e2=fit_ellipse(across[sel],area)
    centre=disc_centre + offset

    m[PA]=0
    box, t_grid, inside = footprint_grid(m.shape, spacing, centre, normal, e1, e2, A, B, RECESS_MM)
    filled = shape_opening(m, forbidden, spacing, box, t_grid, inside, LABELS["RV"], LABELS["PA"], PA)

    quality, flags = opening_quality(filled, A, B)
    info = {**opening_record(img, centre, normal, e1, e2, A, B), **quality,
            "trunk_axis_world": to_world_vector(img, trunk / np.linalg.norm(trunk)),
            "angle_valve_plane_to_trunk_deg": round(float(np.degrees(np.arccos(abs(normal @ trunk) / np.linalg.norm(trunk)))), 1)}
    info["status"] = ";".join(flags) if flags else "ok"
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_dir", type=Path, required=True)
    args = parser.parse_args()

    img = sitk.ReadImage(str(args.case_dir / "mask_rh.nii.gz"))
    m = sitk.GetArrayFromImage(img)
    spacing = np.array(img.GetSpacing()[::-1])                                # (z, y, x)
    forbidden = forbidden_mask(args.case_dir)

    report = {"rules": {"plug_mm": PLUG_MM, "recess_mm": RECESS_MM, "near_shell_mm": NEAR_SHELL_MM,
                        "far_shell_mm": FAR_SHELL_MM, "section_window_mm": SECTION_WINDOW_MM}}
    report["SVC"] = vein_opening(m, forbidden, spacing, img, "SVC")
    report["IVC"] = vein_opening(m, forbidden, spacing, img, "IVC")
    report["RVOT"] = rvot_opening(m, forbidden, spacing, img, args.case_dir)

    # whatever of the vessels survived outside the plugs (bits beyond the footprints) goes
    plug_pieces = {}
    for name, label in (("SVC", 4), ("IVC", 5), ("PA", 3)):
        pieces, n = ndi.label(m == label, STRUCT)
        if n > 1:
            near_chamber = ndi.binary_dilation((m == 1) | (m == 2), STRUCT)
            keep = [k for k in range(1, n + 1) if (near_chamber & (pieces == k)).any()]
            m[(pieces > 0) & ~np.isin(pieces, keep)] = 0
        plug_pieces[name] = int(ndi.label(m == label, STRUCT)[1])
    pieces, n = ndi.label(m > 0, STRUCT)
    if n > 1:                                                               # one connected blood pool
        sizes = ndi.sum(np.ones_like(pieces), pieces, index=range(1, n + 1))
        drop = (pieces > 0) & (pieces != int(np.argmax(sizes)) + 1)
        report["removed_fragments_ml"] = round(float(drop.sum()) * float(np.prod(spacing)) / 1000.0, 3)
        m[drop] = 0
    report["union_components"] = int(ndi.label(m > 0, STRUCT)[1])
    report["plug_pieces"] = plug_pieces
    report["volumes_ml"] = {k: round(float((m == v).sum()) * float(np.prod(spacing)) / 1000.0, 2)
                            for k, v in (("RA", 1), ("RV", 2))}

    problems = [f"{k}:{report[k]['status']}" for k in ("SVC", "IVC", "RVOT") if report[k]["status"] != "ok"]
    if report["union_components"] != 1:
        problems.append(f"union_components:{report['union_components']}")
    problems += [f"{k}_plug_pieces:{v}" for k, v in plug_pieces.items() if v != 1]
    report["problems"], report["included"] = problems, not problems

    out = sitk.GetImageFromArray(m)
    out.CopyInformation(img)
    sitk.WriteImage(out, str(args.case_dir / "mask_rh_cut.nii.gz"), useCompression=True)
    (args.case_dir / "orifices.json").write_text(json.dumps(report, indent=2))
    print(f"{args.case_dir.name}: included={report['included']} {problems}")
    for k in ("SVC", "IVC", "RVOT"):
        r = report[k]
        print(f"   {k:4s} {r['status']:20s} " + "  ".join(f"{q}={r[q]}" for q in
              ("equivalent_diameter_mm", "semi_axes_mm", "recess_fill_depth_mm", "angle_valve_plane_to_trunk_deg") if q in r))
