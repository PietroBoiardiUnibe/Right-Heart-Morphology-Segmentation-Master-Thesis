"""
Step 05b replaces the previous 05 for the physician-corrected manual segmentations: where each opening is and which way its vessel points,
measured on the label map of 04b. No plane and NO ellipse are fitted whatsoever to preserve anatomical fidelity: the opening itself is the patch 
onto chamber wall that touches the vessel label, found and outlined (rim) on the surface by 06/06b.
This step only returns opening with centre and a vessel direction, used to label and order the rims.

Openings:   SVC (RA|SVC), IVC (RA|IVC), RVOT (RV|PA, pulmonary valve level), TV (RA|RV, tricuspid orifice).
Centre:  mean centre of the voxel faces shared by the two labels (boundaries identified and validated)
Axis:  direction of the vessel stub: centroid of the 'beyond' label within SHELL_MM, minus the centre (chamber -> vessel; for the TV RA -> RV).
Measured on the mask, it is independent of how flat the junction is in reality
Missing: no shared face: status 'no_contact' (e.g. a wrongly placed SVC label) -> case excluded upstream.

Inputs : <case_dir>/mask_rh.nii.gz (04b).     Outputs: <case_dir>/orifices.json
Usage  : python 05b_interface_orifices.py --cases_dir <cases folder> [--ids ...]
"""

import argparse
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

LABELS={"RA":1,"RV":2,"PA":3,"SVC":4,"IVC":5}
OPENINGS={"SVC":("RA","SVC"),"IVC":("RA","IVC"),"RVOT":("RV","PA"),"TV":("RA","RV")} # name: (chamber, beyond)
SHELL_MM=(2.0,10.0) # part of the vessel (beyond label) used for its direction, distance from the chamber
CROP_MM=20.0 # work in a box this far around the interface (keeps the distance map small)

def interface_faces(m,a,b):
    """Centres (voxel index, array order z y x) of all faces shared by labels a and b"""
    centres=[]

    for axis in range(3):
        lo=[slice(None)]*3; hi=[slice(None)]*3
        lo[axis],hi[axis]=slice(0,-1),slice(1,None)
        A,B=m[tuple(lo)], m[tuple(hi)]
        idx=np.argwhere(((A==a)&(B==b)) | ((A==b) & (B==a))).astype(float)
        idx[:,axis]+=0.5   # face centre between the two voxels
        centres.append(idx)

    return np.vstack(centres)

def to_world(img,idx_zyx):
    sp=np.array(img.GetSpacing()); D=np.array(img.GetDirection()).reshape(3,3)
    return np.array(img.GetOrigin()) + (idx_zyx[:,::-1]*sp) @ D.T

def measure(img,m,a,b):
    faces=interface_faces(m,a,b)
    if len(faces)==0:
        return{"status":"no_contact"}
    sp=np.array(img.GetSpacing())[::-1]

    lo=np.maximum(np.floor(faces.min(axis=0) - CROP_MM/sp).astype(int),0)
    hi=np.minimum(np.ceil(faces.max(axis=0)+ CROP_MM/sp).astype(int)+1,m.shape)
    box=m[lo[0]:hi[0],lo[1]:hi[1],lo[2]:hi[2]]
    dist=ndi.distance_transform_edt(box!=a, sampling=sp)   # mm from the chamber
    stub=np.argwhere((box==b) & (dist>= SHELL_MM[0]) & (dist <= SHELL_MM[1])) + lo
    c=to_world(img, faces).mean(axis=0)

    if len(stub)==0:
        return {"status":"no_stub","centre_world":c.round(3).tolist(), "faces": int(len(faces))}
    axis=to_world(img,stub.astype(float)).mean(axis=0)-c

    return {"status":"interface","centre_world":c.round(3).tolist(),
            "normal_world":(axis/np.linalg.norm(axis)).round(5).tolist(),   # vessel direction, used by 06 and 06b
            "faces":int(len(faces)),"stub_voxels":int(len(stub))}

def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases_dir", type=Path, required=True)
    ap.add_argument("--ids", nargs="*")
    args=ap.parse_args()

    for case in sorted(p for p in args.cases_dir.iterdir() if (p/"mask_rh.nii.gz").exists()):
        if args.ids and case.name not in args.ids:
            continue
        img=sitk.ReadImage(str(case/"mask_rh.nii.gz"))
        m=sitk.GetArrayFromImage(img)
        report={name: measure(img,m, LABELS[a],LABELS[b]) for name ,(a,b) in OPENINGS.items()}
        report["source"] = "05b: physician label interfaces (centre) + vessel stub (direction)"
        (case/"orifices.json").write_text(json.dumps(report,indent=2))
        print(case.name, " | ".join(f"{k} {v['status']}" + (f" ({v['faces']} faces)" if "faces" in v else "")
                                    for k, v in report.items() if k != "source"), flush=True)

if __name__=="__main__":
    main()


