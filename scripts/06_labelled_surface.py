"""
Step 6 of the right-heart pipeline: closed, labelled surface of the UNCUT chambers (RA + RV with its outflow tract)
for the SSM. Nothing is filled, trimmed or moved: the SSM sees only segmented chamber wall.

1. surface of the chambers of mask_rh (04): Gaussian smoothing in mm, isotropic resampling, marching cubes,
   vertices in WORLD coordinates (mm, direction applied)
2. isotropic remeshing to uniform triangles of TARGET_EDGE_MM, triangles turned outwards
3. footprints = vessel entry zones, labelled on the wall (patch 10 SVC, 11 IVC, 12 RVOT): the cap at the chamber
   tip that lies inside the vessel lumen (ellipse measured by 05 on the vessel) and faces the vessel (smoothed
   normal within MAX_TILT_DEG of the vessel axis), grown from the tip. 05 is used as a MEASUREMENT only.
4. per-vertex labels: chamber (1 RA, 2 RV), patch, TV annulus band (within TV_BAND_MM of both RA and RV), and the
   smooth footprint score fp_<vessel> (<= 1 inside; its 1-isoline is the rim used later as a contour domain)
5. QC: closed, one piece, manifold, genus 0, no self-intersections, triangle quality, every footprint present,
   one piece with one rim

Inputs (case folder): mask_rh.nii.gz, orifices.json.     Outputs: surface_chambers.vtp, surface_chambers_qc.json
"""

import argparse
import json
from pathlib import Path
import numpy as np
import vtk
import pymeshlab
import SimpleITK as sitk

from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import marching_cubes
from vtkmodules.util import numpy_support as nps

CHAMBERS={"RA":1,"RV":2}
OPENINGS={"SVC":10,"IVC":11,"RVOT":12} #patch id
SIGMA_MM=1.0 # Gaussin smoothing parameter for the chamber mask before marching cubes renders surface
ISO_MM=0.75 #isotropic grid for marching cubes
TARGET_EDGE_MM=1.5 #edge length for triangulation after meshing, striking a balance
TV_BAND_MM=1.5 # vertices within this distance of both RA and RV form the annulus band to be preserved
MIN_ANGLE_DEG=15.0  # more than 0.1 % of triangle angles below this: flag (curvature QC)
MAX_TILT_DEG=65.0 # wall facing the vessel: smoothed normal within this angle of the vessel axis
NORMAL_SMOOTH_IT=3 # neighbour averaging of the vertex normals (removes triangle-scale noise)
SCORE_SMOOTH_IT=2  # neighbour averaging of the footprint score (smooth rim)
# surface in world coordinates (the openings in orifices.json are already in world coordinates, so the surface must be too),
#re-mesh and then orientation check

def chamber_surface(mask_img:sitk.Image)->tuple[np.ndarray,np.ndarray]:
    """Smoothed marching cubes surface of RA+RV as world vertices (x,y,z,mm) + triangles"""
    m=sitk.GetArrayFromImage(mask_img)
    binary=sitk.GetImageFromArray(np.isin(m,list(CHAMBERS.values())).astype(np.float32))
    binary.CopyInformation(mask_img)

    #crop to chamber + margin so that smoothing and resampling stay cheap
    idx=np.nonzero(sitk.GetArrayViewFromImage(binary))
    pad=[int(np.ceil(4*SIGMA_MM/s)) + 2 for s in mask_img.GetSpacing()[::-1]]
    lo_zyx=[max(int(i.min()) - p,0) for i , p in zip(idx,pad)]
    hi_zyx=[min(int(i.max()) +p+1,n) for i,p,n in zip(idx,pad,m.shape)]
    roi=sitk.RegionOfInterest(binary, [h - l for h,l in zip(hi_zyx,lo_zyx)][::-1], lo_zyx[::-1])
    smooth=sitk.SmoothingRecursiveGaussian(roi,SIGMA_MM)  # sigma in mm (uses the spacing)
    size=[int (np.ceil(n*s/ISO_MM)) for n,s in zip(smooth.GetSize(),smooth.GetSpacing())]
    iso=sitk.Resample(smooth,size,sitk.Transform(),sitk.sitkLinear,smooth.GetOrigin(),[ISO_MM]*3,
    smooth.GetDirection(),0.0)
    verts_zyx,faces,_,_=marching_cubes(sitk.GetArrayFromImage(iso),level=0.5,spacing=(ISO_MM,)*3)
    direction=np.array(iso.GetDirection()).reshape(3,3)
    world=np.array(iso.GetOrigin()) + verts_zyx[:,::-1] @ direction.T   # (x,y,z) offsets rotated by the direction matrix
    return world, faces.astype(np.int64)

def remesh(points:np.ndarray, faces:np.ndarray)->tuple[np.ndarray,np.ndarray]:
    """Isotropic re-meshing to obtain uniform triangles, surface kept within a quarter edge of the input"""
    ms=pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=points,face_matrix=faces))
    ms.meshing_isotropic_explicit_remeshing(targetlen=pymeshlab.PureValue(TARGET_EDGE_MM),iterations=10,
    checksurfdist=True,maxsurfdist=pymeshlab.PureValue(0.25*TARGET_EDGE_MM))
    mesh=ms.current_mesh()
    return mesh.vertex_matrix().astype(np.float64), mesh.face_matrix().astype(np.int64)

def orient_outwards(points: np.ndarray, faces: np.ndarray)-> np.ndarray:
    """ Make the enclosed volume positive (triangles counter-clockwise seen from outside)
    ImageCAS's flipped y-axis mirrors the frame, which reverses the marching-cubes winding 
    """
    a,b,c= points[faces[:,0]], points[faces[:,1]],points[faces[:,2]]
    return faces if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() > 0 else faces[:, ::-1].copy()

def vertex_normals(points,faces)->np.ndarray:
    """Area weighted outward vertex normals (faces already orient outwards)"""
    fn = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    vn=np.zeros_like(points)
    for k in range(3):
        np.add.at(vn,faces[:,k],fn)
    return vn/np.maximum(np.linalg.norm(vn,axis=1,keepdims=True),1e-12)

def vertex_neighbours(faces, n_points):
    """Sparse 0/1 matrix: vertices connected by an edge"""
    e=np.vstack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    a = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_points, n_points))
    return ((a + a.T) > 0).astype(float).tocsr()

def smooth_on_mesh(neighbours, values, iterations):
    """Replace each value by the mean of itself and its neighbours, "iterations" times """
    count=np.asarray(neighbours.sum(axis=1)).ravel() + 1.0
    for _ in range(iterations):
        values=(neighbours @ values + values)/ (count[:, None] if values.ndim==2 else count)
    return values

def label_footprints(points,faces,openings)->tuple[np.ndarray, dict,dict]:
    """
    Footprint of each vessel = cap at the chamber tip facing the vessel; score per vertex
     max(rho, tilt / MAX_TILT_DEG): rho = elliptic radius in the vessel's lumen ellipse (1 = its outline), tilt =
    angle between the smoothed wall normal and the vessel axis. Score smoothed on the mesh; the footprint is the
    part with score <= 1 connected to the tip (most forward wall vertex inside the ellipse)
    """
    neighbours=vertex_neighbours(faces,len(points))
    normals = smooth_on_mesh(neighbours, vertex_normals(points,faces), NORMAL_SMOOTH_IT)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    face_patch=np.zeros(len(faces), np.int32)
    info,scores={},{}

    for name, pid in OPENINGS.items():
        o=openings[name]
        if "centre_world" not in o:                               # 05 could not measure this opening
            info[name]= {"opening": o.get("status", "missing")}
            continue
        c,n,e1,e2=(np.array(o[k]) for k in ("centre_world", "normal_world", "axis1_world", "axis2_world"))
        A,B=o["semi_axes_mm"]
        rel=points - c
        rho=np.sqrt(((rel @ e1) / A) ** 2 + ((rel @ e2) / B) ** 2)
        tilt=np.degrees(np.arccos(np.clip(normals @ n, -1.0, 1.0)))
        score=smooth_on_mesh(neighbours,np.maximum(rho,tilt/MAX_TILT_DEG), SCORE_SMOOTH_IT)
        scores[f"fp_{name}"] = score.astype(np.float32)
        inside=score<=1.0
        candidates=np.nonzero(inside)[0]

        if len(candidates)==0:
            info[name]={"opening": o["status"], "tip_found": False}
            continue
        tip=candidates[np.argmax(rel[candidates] @ n)]  # most forward wall vertex towards the vessel
        _,component =connected_components(neighbours[inside][:,inside], directed=False)
        keep = np.zeros(len(points), bool)
        keep[candidates[component == component[np.searchsorted(candidates, tip)]]] = True
        selected=keep[faces].all(axis=1) & (face_patch==0)
        face_patch[selected]=pid
        info[name] = {"opening": o["status"], "tip_found": True, "opening_area_mm2": round(float(np.pi * A * B), 1),
                      "tip_beyond_plane_mm": round(float(rel[tip] @ n), 2)}

    return face_patch, info, scores

def boundary_loops(faces, selected)-> int:
    """
    Number of closed boundary curves of a face patch (a disc-like footprint has exactly 1)
    """
    sub=faces[selected]

    if len(sub)==0:
        return 0

    edges= np.sort(np.vstack([sub[:, [0, 1]], sub[:, [1, 2]], sub[:, [2, 0]]]), axis=1)
    uniq, count=np.unique(edges,axis=0, return_counts=True)
    rim=uniq[count==1] # edges used by one patch face only

    if len(rim)==0:
        return 0

    nodes,inv = np.unique(rim, return_inverse=True)
    inv = inv.reshape(-1, 2)
    graph=coo_matrix((np.ones(len(rim)), (inv[:, 0], inv[:, 1])), shape=(len(nodes), len(nodes)))
    return int(connected_components(graph, directed=False)[0])




def world_to_index(img:sitk.Image, points: np.ndarray)-> np.ndarray:
    """ World (x,y,z) -> voxel index (z,y,x), rounded and kept inside the image"""
    direction=np.array(img.GetDirection()).reshape(3,3)
    ijk=((points-np.array(img.GetOrigin()))@direction)/np.array(img.GetSpacing())
    return np.clip(np.round(ijk[:,::-1]).astype(int),0,np.array(img.GetSize()[::-1]) -1)

def vertex_labels(points,faces,face_patch, mask_img)-> dict:
    m=sitk.GetArrayFromImage(mask_img)
    chamber=np.where(np.isin(m,list(CHAMBERS.values())),m,0)
    spacing=np.array(mask_img.GetSpacing()[::-1])
    idx=tuple(world_to_index(mask_img,points).T)
    _,nearest=ndi.distance_transform_edt(chamber==0,return_indices=True)
    chamber_of_vertex=chamber[tuple(nearest)][idx].astype(np.int32) #nearest RA/RV voxel
    d_ra=ndi.distance_transform_edt(chamber != CHAMBERS["RA"], sampling=spacing)[idx]
    d_rv=ndi.distance_transform_edt(chamber != CHAMBERS["RV"],sampling=spacing)[idx]
    tv=((d_ra<=TV_BAND_MM) & (d_rv<=TV_BAND_MM)).astype(np.int32)
    vertex_patch=np.zeros(len(points),np.int32)

    for pid in OPENINGS.values():
        vertex_patch[np.unique(faces[face_patch==pid])]=pid  # rim vertices belong to the footprint
    face_chamber=np.round(chamber_of_vertex[faces].mean(axis=1)).astype(np.int32)

    return {"point": {"chamber": chamber_of_vertex, "patch": vertex_patch, "tv_annulus": tv},
            "cell": {"chamber": face_chamber, "patch": face_patch}}

# QC : checks that keep failure modes of meshing out of SSM pipeline (folds, spikes in warping, broken footprints)

def mesh_qc(points, faces, labels, footprint_info, openings)->dict:
    ms=pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=points,face_matrix=faces))
    topo=ms.get_topological_measures()
    ms.compute_selection_by_self_intersections_per_face()
    a,b,c=points[faces[:,0]],points[faces[:,1]],points[faces[:,2]]
    face_area=0.5 * np.linalg.norm(np.cross(b-a,c-a), axis=1)
    edges=np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edge_len=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]], axis=1)
    angles=[]
    for i in range(3):
        u=points[faces[:,(i+1)%3]]-points[faces[:,i]]
        v=points[faces[:,(i+2)%3]]-points[faces[:,i]]
        angles.append(np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", u, v)
                                                   / (np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)), -1, 1))))
    
    angles=np.concatenate(angles)

    qc={"vertices":len(points),"faces":len(faces),"boundary_edges":int(topo["boundary_edges"]),
        "components":int(topo["connected_components_number"]),"two_manifold":bool(topo["is_mesh_two_manifold"]),
        "genus": int(topo["genus"]), "self_intersecting_faces":int(ms.current_mesh().selected_face_number()),
        "edge_mm_mean": round(float(edge_len.mean()),3),"edge_cv":round(float(edge_len.std()/edge_len.mean()),3),
        "min_angle_deg": round(float(angles.min()), 2),
          "angles_below_min_pct": round(float((angles < MIN_ANGLE_DEG).mean() * 100), 3),
          "max_valence": int(np.bincount(faces.ravel()).max()),
          "volume_ml": round(float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6000.0), 2),
          "tv_annulus_vertices": int(labels["point"]["tv_annulus"].sum()), "footprints": {}}
    
    face_vec=0.5*np.cross(b-a,c-a) #face normal x area

    for name, pid in OPENINGS.items():
        sel=labels["cell"]["patch"]==pid
        fp={"faces": int(sel.sum()), "area_mm2": round(float(face_area[sel].sum()), 1),
              "boundary_loops": boundary_loops(faces, sel), **footprint_info[name]}
        if "opening_area_mm2" in fp: # footprint seen along the vessel axis vs lumen
            n=np.array(openings[name]["normal_world"])
            fp["projected_area_ratio"] = round(float(abs((face_vec[sel] @ n).sum()) / fp["opening_area_mm2"]), 3)
        qc["footprints"][name]=fp

    checks={"not_closed":qc["boundary_edges"]>0, "several_pieces":qc["components"]!=1,
            "non_manifold": not qc["two_manifold"],"genus_not_0": qc["genus"] != 0, "self_intersections":qc["self_intersecting_faces"]>0,
            "thin_triangles":qc["angles_below_min_pct"]>0.1,"inside_out":qc["volume_ml"]<=0, "no_tv_annulus": qc["tv_annulus_vertices"] == 0
            }

    for name, fp in qc["footprints"].items():
        checks[f"{name}_footprint_missing"]=fp["faces"]==0
        checks[f"{name}_footprint_not_disc"]=fp["faces"]>0 and fp["boundary_loops"]!=1
    qc["problems"] = [k for k, failed in checks.items() if failed]
    qc["included"] = not qc["problems"]

    return qc

#writing VTP mesh reference

def write_vtp(points,faces,labels,path:Path)->None:
    poly=vtk.vtkPolyData()
    vpts=vtk.vtkPoints()
    vpts.SetData(nps.numpy_to_vtk(np.ascontiguousarray(points),deep=True))
    poly.SetPoints(vpts)
    cells=vtk.vtkCellArray()
    cells.SetCells(len(faces),nps.numpy_to_vtkIdTypeArray(
        np.hstack([np.full((len(faces),1),3, np.int64),faces]).ravel(), deep=True))
    poly.SetPolys(cells)

    for where, arrays in labels.items():
        data=poly.GetPointData() if where == "point" else poly.GetCellData()

        for name, values in arrays.items():
            arr=nps.numpy_to_vtk(np.ascontiguousarray(values),deep=True)
            arr.SetName(name)
            data.AddArray(arr)

    note=vtk.vtkStringArray()
    note.SetName("description")
    note.InsertNextValue("mm, world LPS; chamber 1 RA 2 RV; patch 0 wall 10 SVC 11 IVC 12 RVOT footprint (vessel entry zone on the uncut wall); tv_annulus 0/1; fp_<vessel> footprint score (rim = isoline 1)")
    poly.GetFieldData().AddArray(note)
    writer=vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    writer.SetDataModeToBinary()
    writer.Write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case_dir", type=Path, required=True)
    args = parser.parse_args()

    mask_img = sitk.ReadImage(str(args.case_dir / "mask_rh.nii.gz"))
    openings = json.loads((args.case_dir / "orifices.json").read_text())   # 05 used as MEASUREMENT only
    
    points, faces = chamber_surface(mask_img)
    points, faces = remesh(points, faces)
    faces = orient_outwards(points, faces)
    face_patch, footprint_info, scores=label_footprints(points, faces, openings)
    labels = vertex_labels(points, faces, face_patch, mask_img)
    labels["point"].update(scores)
    qc = mesh_qc(points, faces, labels, footprint_info, openings)

    write_vtp(points, faces, labels, args.case_dir / "surface_chambers.vtp")
    (args.case_dir / "surface_chambers_qc.json").write_text(json.dumps(qc, indent=2))
    print(f"{args.case_dir.name}: included={qc['included']} {qc['problems']}")
    print(f"   {qc['vertices']} vertices, edge {qc['edge_mm_mean']} mm (cv {qc['edge_cv']}), min angle {qc['min_angle_deg']} deg, "
          f"max valence {qc['max_valence']}, volume {qc['volume_ml']} mL, genus {qc['genus']}, "
          f"self-intersections {qc['self_intersecting_faces']}, tv band {qc['tv_annulus_vertices']} vertices")
    for name, fp in qc["footprints"].items():
        print(f"   {name}: {fp}")

    