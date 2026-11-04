"""
Step 06b: the rims pf the openings as 3D curves on the chamber wall directly (no plane), resampled to a fixed number of points in correspondence
across hearts, for the SSM (they join the shape vector) and for cutting the holes in the synthetic hearts.

RIM:  SVC, IVC, RVOT: boundary of the footprint patch of 06 (wall touching the vessel label).
             TV: boundary between RA wall and RV wall on the merged surface (where the two labels meet = annulus).
             Ordered as one closed loop of mesh vertices (the longest loop if there were several).

POINTS: the loop is smoothed along itself (moving average of WINDOW vertices, removes the mesh-edge zig-zag) and put back onto the wall.
             Seen along the vessel axis from the opening centre, point j is where the rim crosses th direction j*360/K deg counter-clockwise from an
             anatomical reference - towards the TV centre for SVC/IVC/RVOT, towards the RVOT centre for the TV. Fixed angles about a fixed anatomical frame=
             the same point number means the same anatomical side in every heart.
LANDMARKS: per opening: centre, tip = centre + TIP_MM x vessel axis (orientation), K rim points.

DESCRIPTORS: perimeter, vector area of the loop (= area seen along its mean normal) and equivalent diameter, non-planarity
(RMS and peak-to-peak distance of the rim from its best plane: the saddle).

Inputs : <case_dir>/surface_chambers.vtp (06, interface footprints), orifices.json (05b)
Outputs: <case_dir>/rims.json, rims_qc.png
Usage  : python 06b_rims.py --cases_dir <cases folder> [--ids ...] [--k_vessel 16] [--k_tv 32]  
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from vtkmodules.util.numpy_support import vtk_to_numpy

PATCH={"SVC":10,"IVC":11,"RVOT":12}
REFERENCE={"SVC":"TV","IVC":"TV","RVOT":"TV","TV":"RVOT"}
TIP_MM=10.0  # orientation landmark: centre + TIP_MM x vessel axis (the PCA learns the vessel direction from it)
COLOURS = {"SVC": "#eb6834", "IVC": "#7a3fb0", "RVOT": "#1baf7a", "TV": "#2a78d6"}

#for vessels, an edge used exactly by one footprint face only lies on the border

def read_surface(path):
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(path)); r.Update()
    s=r.GetOutput()
    P=vtk_to_numpy(s.GetPoints().GetData()).astype(float)
    F=vtk_to_numpy(s.GetPolys().GetConnectivityArray()).reshape(-1,3)
    return P,F,vtk_to_numpy(s.GetCellData().GetArray("patch")), vtk_to_numpy(s.GetCellData().GetArray("chamber"))

def loops(edges):
    """closed loops of a set of boundary edges, each as an ordered vertex list; longest is first"""
    nodes, inv=np.unique(edges, return_inverse=True); inv=inv.reshape(-1,2)
    graph=coo_matrix((np.ones(len(inv)), (inv[:, 0], inv[:, 1])), shape=(len(nodes), len(nodes)))
    graph=((graph+graph.T)>0).tocsr()
    n_comp,comp=connected_components(graph, directed=False)
    out=[]

    for k in range (n_comp):
        members=np.nonzero(comp==k)[0]
        start, order, prev=members[0],[members[0]],-1
        while True:# walk around the loop
            nxt=[j for j in graph.indices[graph.indptr[order[-1]]:graph.indptr[order[-1] + 1]] if j != prev]
            if not nxt or nxt[0]== start:
                break
            prev=order[-1]; order.append(nxt[0])
            if len(order)>len(members):
                break
        out.append(nodes[order])

    return sorted(out, key=len, reverse=True)

def boundary_edges(faces,selected,other=None):
    """edges used by exactly one selected face; with "other" m edges shared between 
    a selected and an "other" face """
    def edge_list(sub):
        return np.sort(np.vstack([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]]), axis=1)
    if other is None:
        e,cnt=np.unique(edge_list(faces[selected]),axis=0, return_counts=True)
        return e[cnt==1]
    a,b={tuple(x) for x in edge_list(faces[selected])}, {tuple(x) for x in edge_list(faces[other])}
    return np.array(sorted(a & b))

#smoothing and fixed-angle sampling , the core of correspondence

def smooth_loop(X,surface_locator, window=5):
    """moving average along the closed loop (removing the mesh-edge zig-zag artifact), then back onto the wall of the chamber"""
    h=window//2
    Y=np.mean([np.roll(X,k,axis=0) for k in range(-h,h+1)],axis=0)
    out=np.empty_like(Y)
    for i, y in enumerate(Y):
        cp=[0.0,0.0,0.0]; cid=vtk.reference(0); sub=vtk.reference(0); d2=vtk.reference(0.0)
        surface_locator.FindClosestPoint(y,cp,cid,sub,d2); out[i]=cp

    return out

def resample(X,centre, axis, reference,k):
    """ K rim points ar fixed angles around the opening centre: point j is where the rim, seen along the vessel axis,
     crosses the direction rotated by j*360/K from the anatomical reference (counter-clockwise about the axis)
     Returns the points and the number of directions that crossed the rim more than once (0= star-shaped rim we wish to have)
    """
    ref=reference-centre; ref-=(ref@axis)*axis; ref/=np.linalg.norm(ref)
    side=np.cross(axis,ref)
    uv=np.column_stack([(X-centre)@ref, (X-centre)@side])
    a,b=uv, np.roll(uv,-1,axis=0)
    pts, multi=[],0

    for th in 2*np.pi*np.arange(k)/k:
        d=np.array([np.cos(th),np.sin(th)])
        e=b-a
        den=d[0] * e[:, 1] - d[1] * e[:, 0]                                       # ray x segment
        with np.errstate(divide="ignore",invalid="ignore"):
            t=(a[:,0]*e[:,1]-a[:,1]*e[:,0])/den# distance along the ray
            u = (a[:, 0] * d[1] - a[:, 1] * d[0]) / den     
        hit=np.nonzero((t>0)&(u>=0) & (u<1))[0]
        if len(hit)==0:
            return None,k
        multi+=len(hit)>1
        j=hit[np.argmax(t[hit])]# outermost crossing
        pts.append(X[j]+u[j]*(np.roll(X,-1,axis=0)[j] - X[j]))
    return np.array(pts), multi


def descriptors(pts):
    c=pts.mean(axis=0) #rim centroid
    va=0.5 * np.cross(pts-c, np.roll(pts, -1,axis=0)-c).sum(axis=0) # vector area of the closed rim
    area=float(np.linalg.norm(va))
    _,sv,Vt=np.linalg.svd(pts-c,full_matrices=False)
    off=(pts-c)@Vt[2]

    return {"rim_centroid":c.round(3).tolist(),"vector_area_mm2":round(area,1),"eq_diameter_mm":round(2*np.sqrt(area/np.pi),2),
    "nonplanar_rms_mm": round(float(np.sqrt(np.mean(off ** 2))),2), "nonplanar_pp_mm":round(float(np.ptp(off)),2)
    }

def rims_of_case(case,k_vessel, k_tv):
    P,F,patch,chamber=read_surface(case/"surface_chambers.vtp")
    r=vtk.vtkXMLPolyDataReader(); r.SetFileName(str(case/"surface_chambers.vtp")); r.Update()
    locator=vtk.vtkCellLocator(); locator.SetDataSet(r.GetOutput()); locator.BuildLocator()
    ori=json.loads((case/"orifices.json").read_text())
    out={}

    for name in ("SVC","IVC","RVOT","TV"):
        o=ori.get(name,{})
        if o.get("status")!="interface" or REFERENCE[name] not in ori or "centre_world" not in ori[REFERENCE[name]]:
            out[name]={"status":"missing"}; continue
        if name=="TV":
            edges=boundary_edges(F,(chamber==1) & (patch==0), (chamber==2) & (patch==0))
        else:
            edges=boundary_edges(F,patch==PATCH[name])
        if len(edges)==0:
            out[name]={"status":"no_rim"}; continue
        found=loops(edges)
        centre,axis=np.array(o["centre_world"]),np.array(o["normal_world"])
        X=smooth_loop(P[found[0]],locator)
        pts, multi = resample(X, centre, axis, np.array(ori[REFERENCE[name]]["centre_world"]), k_tv if name == "TV" else k_vessel)
        if pts is None:
            out[name]={"status":"centre_outside_rim"}; continue
        out[name] = {"status": "ok", "points": pts.round(3).tolist(), "centre": centre.round(3).tolist(),
                     "tip": (centre + TIP_MM * axis).round(3).tolist(), "multi_crossings": int(multi),
                     "perimeter_mm": round(float(np.linalg.norm(np.roll(X, -1, axis=0) - X, axis=1).sum()), 1),
                     "loops_found": len(found), "loop_vertices": int(len(found[0])), **descriptors(pts)}
    return out, (P,F,patch, ori)

def qc_figure(path, case, rims, P, F, patch, ori):
    tv = np.array(ori["TV"]["normal_world"]); c = np.array(ori["TV"]["centre_world"])
    e1 = np.array(ori["RVOT"]["centre_world"]) - c; e1 -= (e1 @ tv) * tv; e1 /= np.linalg.norm(e1); e2 = np.cross(tv, e1)
    views = ((e1, e2, "seen along the TV axis (RA -> RV)"), (e1, tv, "side view (RV up)"), (e2, tv, "side view, rotated 90 deg"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    sub = np.random.default_rng(0).choice(len(P), min(len(P), 12000), replace=False)
    for ax, (u, v, title) in zip(axes, views):
        Q = P[sub] - c
        ax.scatter(Q @ u, Q @ v, s=0.3, color="#c8c7c2")
        for name, pid in PATCH.items():
            pts = P[np.unique(F[patch == pid])] - c
            ax.scatter(pts @ u, pts @ v, s=0.6, color=COLOURS[name], alpha=0.35)
        for name, r in rims.items():
            if r["status"] != "ok":
                continue
            R = np.array(r["points"]) - c
            ax.plot(np.append(R @ u, (R @ u)[0]), np.append(R @ v, (R @ v)[0]), "-", color=COLOURS[name], lw=1.4,
                    label=f"{name} rim d {r['eq_diameter_mm']} mm, saddle {r['nonplanar_pp_mm']} mm")
            ax.plot((R @ u)[0], (R @ v)[0], "o", color=COLOURS[name], ms=5, mec="#0b0b0b")         # point 1
            ax.plot((R @ u)[1], (R @ v)[1], ">", color=COLOURS[name], ms=4, mec="#0b0b0b")         # direction
            o = ori[name]; oc = np.array(o["centre_world"]) - c; ax_ = np.array(o["normal_world"]) * 12
            ax.annotate("", (oc @ u + ax_ @ u, oc @ v + ax_ @ v), (oc @ u, oc @ v), arrowprops=dict(arrowstyle="->", color=COLOURS[name], lw=1.2))
        ax.set_aspect("equal"); ax.tick_params(labelsize=6); ax.set_title(f"{case}: {title}", fontsize=8)
    axes[0].legend(fontsize=6, frameon=False, loc="lower left")
    fig.text(0.5, 0.01, "grey = chamber wall, coloured = footprints (wall touching the vessel label), lines = rims (K points), "
             "circle = point 1 (anatomical start), triangle = running direction, arrows = vessel axes from the mask", ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases_dir", type=Path, required=True)
    ap.add_argument("--ids",nargs="*")
    ap.add_argument("--k_vessel", type=int, default=16)
    ap.add_argument("--k_tv", type=int, default=32)
    args=ap.parse_args()

    for case in sorted(p for p in args.cases_dir.iterdir() if (p/"surface_chambers.vtp").exists()):
        if args.ids and case.name not in args.ids:
            continue
        rims, (P,F,patch,ori)=rims_of_case(case,args.k_vessel,args.k_tv)
        (case/"rims.json").write_text(json.dumps(rims, indent=1))
        qc_figure(case/"rims_qc.png", case.name,rims,P,F, patch, ori)
        print(case.name, " | ".join(f"{k} " + (f"d {v['eq_diameter_mm']} perim {v['perimeter_mm']} saddle {v['nonplanar_pp_mm']} loops {v['loops_found']}"
                                             if v["status"] == "ok" else v["status"]) for k, v in rims.items()), flush=True)

if __name__=="__main__":
    main()




        
