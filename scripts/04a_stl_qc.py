"""
Quality Check of the physician-corrected and validated label surfaces from Inselspital group cardiac CTs (one folder per case, one STL per label)
before any of those enters the SSM pipeline.

Per label: triangles shall be closed (open edges), pieces (connected components), orientation (signed volume < 0 = inward normals to check), volume (ml),
 voxel spacing read from the vertex coordinate grid (staircase step), edge length.

Interfaces: the patch of a chamber surface that touches a neighbor label: (all triangle vertices within TOUCH_MM of the
            neighbour surface): area, flatness (smallest/largest principal spread of its points) and an
            area-equivalent diameter. Interfaces used later: RA|RV (tricuspid orifice), RA|SVC, RA|IVC, RV|PA (pulmonary).
FOV: the scan ends where the labels are cut flat (top = largest z of all labels, bottom = smallest): share of
view  RA / RV vertices lying on that plane (> FOV_PCT % = chamber clipped by the scan).
Flags: label missing, not closed, > 1 piece, interface missing or tiny, chamber clipped by the field of view.

Usage:  python 04a_stl_qc.py --cases_dir <folder with one subfolder per case> --out_csv <file> [--ids ...]

"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

LABELS={"RA":"RA.stl", "RV":"RV.stl", "PA":"Pulmonary Arteries.stl","SVC":"Superior Vena Cava.stl","IVC":"Inferior Vena Cava.stl"}
INTERFACES=(("RA","RV"),("RA","SVC"),("RA","IVC"),("RV","PA"))
TOUCH_MM=0.3 # a chamber triangle belongs to an interface if all its vertices are this close to the neighbour
MIN_INTERFACE_MM2=50.0 #smaller contact = flagged (missing or broken junction)
FOV_PCT=0.5  # chamber vertices on the top / bottom plane of the scan above this share: clipped


def read_stl(path):
    r=vtk.vtkSTLReader(); r.SetFileName(str(path)); r.Update()
    c=vtk.vtkCleanPolyData(); c.SetInputData(r.GetOutput()); c.Update() #merge duplicated stl vertices
    return c.GetOutput()

def arrays(poly):
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(float),vtk_to_numpy(poly.GetPolys().GetConnectivityArray()).reshape(-1,3)

def grid_step(P):
    """Most frequent spacing between distinct vertex coordinates per axis= voxel size of segmentation source"""
    out=[]
    for k in range(3):
        d=np.diff(np.unique(np.round(P[:,k],4)))
        vals,cnt=np.unique(np.round(d[d>1e-3],3),return_counts=True)
        out.append(float(vals[np.argmax(cnt)]))
    return out

def label_qc(poly):
    P,F=arrays(poly)
    fe=vtk.vtkFeatureEdges(); fe.SetInputData(poly)
    fe.BoundaryEdgesOn(); fe.NonManifoldEdgesOn(); fe.FeatureEdgesOff(); fe.ManifoldEdgesOff(); fe.Update()
    cc=vtk.vtkPolyDataConnectivityFilter(); cc.SetInputData(poly);cc.SetExtractionModeToAllRegions(); cc.Update()
    signed=np.einsum("ij,ij->i", P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]])).sum() / 6

    return {"tris":len(F), "open_edges":fe.GetOutput().GetNumberOfCells(), "pieces":cc.GetNumberOfExtractedRegions(), "inward": bool(signed<0), "vol_ml":round(abs(signed)/1000,1),
            "edge_mm":round(float(np.linalg.norm(P[F[:,0]]-P[F[:,1]],axis=1).mean()),2),"voxel_mm":grid_step(P)}

def interface(chamber, neighbour):
    """contact patch of the chamber surface with the neighbour label"""
    P,F=arrays(chamber)
    dist=vtk.vtkImplicitPolyDataDistance(); dist.SetInput(neighbour)
    out=vtk.vtkDoubleArray(); dist.FunctionValue(numpy_to_vtk(np.ascontiguousarray(P),deep=True),out)
    near=np.abs(vtk_to_numpy(out))<TOUCH_MM
    tri=near[F].all(axis=1)
    if tri.sum() <3:
        return {"area_mm2": 0.0, "eq_d_mm": 0.0, "flatness": np.nan}
    area=0.5 * np.linalg.norm(np.cross(P[F[tri, 1]] - P[F[tri, 0]], P[F[tri, 2]] - P[F[tri, 0]]), axis=1).sum()
    C=P[np.unique(F[tri])]
    s=np.linalg.svd(C-C.mean(axis=0),compute_uv=False)
    return {"area_mm2":round(float(area),1),"eq_d_mm":round(float(2*np.sqrt(area/np.pi)),1),"flatness":round(float(s[2]/s[0]),2)}


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases_dir",type=Path,required=True)
    ap.add_argument("--out_csv", type=Path, required=True)
    ap.add_argument("--ids",nargs="*")
    args=ap.parse_args()
    rows=[]

    for case in sorted(p for p in args.cases_dir.iterdir() if p.is_dir()):
        if args.ids and case.name not in args.ids:
            continue
        row, flags, meshes={"case":case.name},[],{}
        for lab, fname in LABELS.items():
            if not (case/fname).exists():
                flags.append(f"{lab}_missing"); continue
            meshes[lab]=read_stl(case/fname)
            q=label_qc(meshes[lab])
            row.update({f"{lab}_{k}": v for k, v in q.items() if k != "voxel_mm"})
            row["voxel_mm"]=q["voxel_mm"]
            if q["open_edges"]:
                flags.append(f"{lab}_open")
            if q["pieces"]>1:
                flags.append(f"{lab}_{q['pieces']}pieces")
        z=np.concatenate([vtk_to_numpy(m.GetPoints().GetData())[:,2] for m in meshes.values()])
        for lab in ("RA","RV"):
            if lab in meshes:
                zl=vtk_to_numpy(meshes[lab].GetPoints().GetData())[:,2]
                clipped = 100 * max(float((zl > z.max() - 0.3).mean()), float((zl < z.min() + 0.3).mean()))
                row[f"{lab}_fov_pct"]=round(clipped,2)
                if clipped>FOV_PCT:
                    flags.append(f"{lab}_fov_clipped")
        for a, b in INTERFACES:
            if a in meshes and b in meshes:
                q=interface(meshes[a],meshes[b])
                row.update({f"{a}|{b}_{k}": v for k, v in q.items()})
                if q["area_mm2"]<MIN_INTERFACE_MM2:
                    flags.append(f"{a}|{b}_interface")
        row["inward_all"] = all(row.get(f"{lab}_inward", False) for lab in LABELS)
        row["flags"] = " ".join(flags)
        rows.append(row)
        print(case.name, row["flags"] or "ok", flush=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
    

    
