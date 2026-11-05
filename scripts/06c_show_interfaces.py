"""
Step 06c (visual QC / thesis figure): how the openings are detected on the SSM input surface from the physicians'
labels, one heart at a time.

Figure      per opening (SVC, IVC, RVOT, TV) two panels, cropped around the rim: looking through the rim along its own
            normal (the direction 06b samples the K points in), and from the side (rim normal up).
            Grey = SSM input wall (06), light colour = the physicians' vessel label (STL), strong colour = footprint
            (wall touching that label), line = rim with its K points (1 = anatomical start, numbers run
            counter-clockwise), star = centre (05b), arrow = vessel axis to the orientation tip (its angle to the rim
            normal is in the title).
ParaView    <case>_interfaces.vtm with blocks: wall (arrays chamber, patch, tv_annulus), rims (polylines, point_id),
            landmarks (centre, tip), label_SVC / label_IVC / label_PA / label_RV (the physicians' STLs).

Inputs : <cases_dir>/<case>/{surface_chambers.vtp, orifices.json, rims.json}, <stl_dir>/<case>/*.stl
Outputs: <cases_dir>/<case>/interfaces_qc.png, <case>_interfaces.vtm (+ its folder)
Usage  : python 06c_show_interfaces.py --cases_dir <cases> --stl_dir <raw STL folder> --ids <case> [...]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

STL = {"SVC": "Superior Vena Cava.stl", "IVC": "Inferior Vena Cava.stl", "RVOT": "Pulmonary Arteries.stl", "TV": "RV.stl"}
PATCH = {"SVC": 10, "IVC": 11, "RVOT": 12}
COLOURS = {"SVC": "#eb6834", "IVC": "#7a3fb0", "RVOT": "#1baf7a", "TV": "#2a78d6"}
CROP_MM = 30.0


def read(path, reader):
    r = reader(); r.SetFileName(str(path)); r.Update()
    return r.GetOutput()


def points_of(poly):
    return vtk_to_numpy(poly.GetPoints().GetData()).astype(float)


def frame(axis, reference_point, centre):
    ref = reference_point - centre; ref -= (ref @ axis) * axis; ref /= np.linalg.norm(ref)
    return ref, np.cross(axis, ref)

def figure(path, case, wall, F, patch, labels, ori, rims):
    P = points_of(wall)
    names = [n for n in ("SVC", "IVC", "RVOT", "TV") if rims[n]["status"] == "ok"]
    fig, axes = plt.subplots(2, len(names), figsize=(4.2 * len(names), 8.4), squeeze=False)
    for j, name in enumerate(names):
        r = rims[name]
        c, axis = np.array(r["rim_centroid"]), np.array(r["rim_normal"])         # view = the way 06b samples the rim
        ref_point=c+(np.array(r["tip"])-np.array(r["centre"])) if r["anchor"]=="tilt" else np.array(ori["RVOT" if name =="TV" else "TV"]["centre_world"])
        ref,side=frame(axis,ref_point,c)
        star, tip = np.array(r["centre"]) - c, np.array(r["tip"]) - c            # 05b centre, orientation tip
        near = lambda X: X[np.linalg.norm(X - c, axis=1) < CROP_MM] - c
        W, L = near(P), near(points_of(labels[name])[::3])
        fp = near(P[np.unique(F[patch == PATCH[name]])]) if name in PATCH else np.empty((0, 3))
        R = np.array(r["points"]) - c
        for i, (u, v, title) in enumerate(((ref, side, "looking through the rim"), (ref, axis, "side view, rim normal up"))):
            ax = axes[i, j]
            if i == 0:                                         # looking into the vessel: draw the far side first
                order = np.argsort(W @ axis); W_ = W[order]
            else:
                W_ = W[np.abs(W @ side) < 3.0]                 # side view: a 6 mm slab through the centre
            ax.scatter(W_ @ u, W_ @ v, s=0.4, color="#b9b8b2")
            L_ = L if i == 0 else L[np.abs(L @ side) < 3.0]
            ax.scatter(L_ @ u, L_ @ v, s=0.4, color=COLOURS[name], alpha=0.15)
            if len(fp):
                f_ = fp if i == 0 else fp[np.abs(fp @ side) < 3.0]
                ax.scatter(f_ @ u, f_ @ v, s=1.2, color=COLOURS[name], alpha=0.8)
            ax.plot(np.append(R @ u, (R @ u)[0]), np.append(R @ v, (R @ v)[0]), "-", color="#0b0b0b", lw=1.0)
            ax.scatter(R @ u, R @ v, s=10, color="#0b0b0b", zorder=3)
            for k in (0, 1, len(R) // 4):
                ax.annotate(str(k + 1), (R[k] @ u, R[k] @ v), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.plot(star @ u, star @ v, "*", color="#d03b3b", ms=10, zorder=4)
            ax.annotate("", (tip @ u, tip @ v), (star @ u, star @ v), arrowprops=dict(arrowstyle="->", color="#d03b3b", lw=1.4))
            ax.set_aspect("equal"); ax.set_xlim(-CROP_MM, CROP_MM); ax.set_ylim(-CROP_MM, CROP_MM); ax.tick_params(labelsize=6)
            ax.set_title(f"{name} - {title}\nd {r['eq_diameter_mm']} mm, saddle {r['nonplanar_pp_mm']} mm, axis {r['axis_vs_rim_deg']} deg off" if i == 0 else f"{name} - {title}", fontsize=8)
    fig.suptitle(f"{case}: openings detected on the SSM input wall from the physicians' labels\n"
                 "grey = wall (06), light colour = physicians' label beyond the opening, strong colour = footprint, "
                 "black = rim (point 1 = anatomical start), star = centre (05b), arrow = vessel axis (orientation tip)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(path, dpi=140); plt.close(fig)

def polyline_block(rims):
    pts, lines, ids, k = vtk.vtkPoints(), vtk.vtkCellArray(), [], 0
    for name, r in rims.items():
        if r["status"] != "ok":
            continue
        R = np.array(r["points"]); n = len(R)
        for x in R:
            pts.InsertNextPoint(*x)
        line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(n + 1)
        for i in range(n + 1):
            line.GetPointIds().SetId(i, k + i % n)
        lines.InsertNextCell(line); ids += list(range(1, n + 1)); k += n
    poly = vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetLines(lines)
    a = numpy_to_vtk(np.array(ids, dtype=np.int32), deep=True); a.SetName("point_id"); poly.GetPointData().AddArray(a)
    return poly


def landmark_block(rims):
    pts, verts = vtk.vtkPoints(), vtk.vtkCellArray()
    for r in rims.values():
        if r["status"] == "ok":
            for x in (r["centre"], r["tip"]):
                verts.InsertNextCell(1); verts.InsertCellPoint(pts.InsertNextPoint(*x))
    poly = vtk.vtkPolyData(); poly.SetPoints(pts); poly.SetVerts(verts)
    return poly


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases_dir", type=Path, required=True)
    ap.add_argument("--stl_dir", type=Path, required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    args = ap.parse_args()
    for cid in args.ids:
        case = args.cases_dir / cid
        wall = read(case / "surface_chambers.vtp", vtk.vtkXMLPolyDataReader)
        F = vtk_to_numpy(wall.GetPolys().GetConnectivityArray()).reshape(-1, 3)
        patch = vtk_to_numpy(wall.GetCellData().GetArray("patch"))
        ori = json.loads((case / "orifices.json").read_text()); rims = json.loads((case / "rims.json").read_text())
        labels = {n: read(args.stl_dir / cid / f, vtk.vtkSTLReader) for n, f in STL.items()}
        figure(case / "interfaces_qc.png", cid, wall, F, patch, labels, ori, rims)
        blocks = {"wall": wall, "rims": polyline_block(rims), "landmarks": landmark_block(rims),
                  "label_SVC": labels["SVC"], "label_IVC": labels["IVC"], "label_PA": labels["RVOT"], "label_RV": labels["TV"]}
        mb = vtk.vtkMultiBlockDataSet(); mb.SetNumberOfBlocks(len(blocks))
        for i, (name, poly) in enumerate(blocks.items()):
            mb.SetBlock(i, poly); mb.GetMetaData(i).Set(vtk.vtkCompositeDataSet.NAME(), name)
        w = vtk.vtkXMLMultiBlockDataWriter(); w.SetFileName(str(case / f"{cid}_interfaces.vtm")); w.SetInputData(mb); w.Write()
        print(f"{cid}: interfaces_qc.png, {cid}_interfaces.vtm", flush=True)


if __name__ == "__main__":
    main()

