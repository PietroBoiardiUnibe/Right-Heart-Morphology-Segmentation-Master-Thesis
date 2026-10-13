"""
Step 6 of the right-heart chambers pipeline to turn the cut mask into closed. labelled surface that can enter 
downstream the SSM pipeline. Only the chambers (RA,RV with its outflow tract towards pulmonary annulus) are meshed

The three opening are already flat in the mask (filled each one up to its plane), so here
they are simply labelled as caps. Nothing artificial is added really as flow extensions are built later+after SSM
directly onto synthetic shapes by growing on top of these caps.

1. surface chambers only, Gaussian smoothing in mm, isotropic resampling, marching cubes;
vertices in WORLD coordinates (mm, direction applied).
2. re-meshing via isotropic triangulation of uniform triangles of TARGET_EDGE_MM
3. caps faces within CAP_TOL_MM of an opening plane and inside its ellipse form a cap.
(vertices are not moved; exact planarity is imposed when flow extensions are built)
4. labels are assigned per vertex: chamber (1 RA, 2 RV), patch (0 wall, 10 SVC inlet, 11 IVC inlet, 12 RVOT outlet), TV annulus
(1 on the band where RA meets segmented RV, namely the tricuspid annulus region)
5. Quality Check controlling closed, manifold, one piece, no self-intersections, triangle quality (orthogonality, skewness in acceptability ranges), caps flat.

Inputs (case folder): mask_rh_cut.nii.gz, orifices.json.     Outputs: surface_rh.vtp, surface_qc.json
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
TARGET_EDGE_MM=1.5 #edge length for triangulation after meshing, striking a balanc
CAP_TOL_MM=0.6 #faces this close to an opening plane (and inside its ellipse) are cap
CAP_RIM_MM=1.0 # ellipse enlarged by this when selecting cap faces
TV_BAND_MM=1.5 # vertices within this distance of both RA and RV form the annulus band to be preserved
CAP_AREA_RANGE=(0.75,1.15)  # cap area / opening ellipse area outside this: flag
MIN_ANGLE_DEG=15.0  # more than 0.1 % of triangle angles below this: flag (curvature QC)

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


# caps are placed, triangles lying on the opening plane and inside its oval.
# only largest connected group is taken

def opening_frame(o:dict):
    return (np.array(o["centre_world"]), np.array(o["normal_world"]),
np.array(o["axis1_world"]), np.array(o["axis2_world"]), *o["semi_axes_mm"])

def largest_face_patch(faces: np.ndarray, selected: np.ndarray)-> np.ndarray:
    """ Keep only largest group of selected faces connected through shared edges (dropping stray faces)"""
    ids=np.nonzero(selected)[0]
    if len(ids)==0:
        return selected
    sub=faces[ids] # every face edge as a sorted vertex pair; two faces sharing an edge are neighbours
    edges=np.sort(np.vstack([sub[:,[0,1]],sub[:,[1,2]], sub[:,[2,0]]]),axis=1)
    owner=np.tile(np.arange(len(ids)),3)
    order=np.lexsort((edges[:,1],edges[:,0])) # identical edges end up next to each other
    e,o=edges[order],owner[order] 
    shared=np.all(e[1:]==e[:-1],axis=1)
    graph = coo_matrix((np.ones(shared.sum()), (o[:-1][shared], o[1:][shared])), shape=(len(ids), len(ids)))
    _,component=connected_components(graph, directed=False)
    out=np.zeros_like(selected)
    out[ids[component==np.bincount(component).argmax()]]=True
    return out

def label_caps(points, faces, openings)-> np.ndarray:
    """ Cap faces per opening: faces within CAP_TOL_MM of the opening plane and inside its ellipse
    (largest connected patch). Vertices are not moved as projecting them onto the plane may fold neighboring wall triangles (self intersections)
    Cap already flat to subvoxel scale, exact planarity is imposed when flow extensions are attached"""
    face_patch=np.zeros(len(faces),np.int32)
    centroids=points[faces].mean(axis=1)

    for name, pid in OPENINGS.items():
        c,n,e1,e2,A,B=opening_frame(openings[name])
        rel=centroids - c
        inside= ((rel @ e1) / (A + CAP_RIM_MM)) ** 2 + ((rel @ e2) / (B + CAP_RIM_MM)) ** 2 <= 1.0
        cap=largest_face_patch(faces, inside & (np.abs(rel @ n) <= CAP_TOL_MM))
        face_patch[cap]=pid
    return face_patch

# each vertex takes the label of nearest RA or RV voxel in the mask and the tricuspid
# annular band is where a vertex is within 1.5 mm of both RA and RV (where the meshes will meet)

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
        vertex_patch[np.unique(faces[face_patch==pid])]=pid  # ring vertices belong to the cap
    face_chamber=np.round(chamber_of_vertex[faces].mean(axis=1)).astype(np.int32)

    return {"point": {"chamber": chamber_of_vertex, "patch": vertex_patch, "tv_annulus": tv},
            "cell": {"chamber": face_chamber, "patch": face_patch}}

# QC : checks that keep failure modes of meshing out of SSM pipeline (folds, spikes in warping, broken caps)

def mesh_qc(points, faces, labels, openings)->dict:
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
          "tv_annulus_vertices": int(labels["point"]["tv_annulus"].sum()), "caps": {}}
    
    for name, pid in OPENINGS.items():
        c0, n, e1, e2, A, B = opening_frame(openings[name])
        sel = labels["cell"]["patch"] == pid
        cap_vertices = np.unique(faces[sel])
        rim = np.intersect1d(cap_vertices, np.unique(faces[~sel]))
        qc["caps"][name] = {"faces": int(sel.sum()),
                            "area_ratio": round(float(face_area[sel].sum() / (np.pi * A * B)), 3),
                            "plane_rms_mm": round(float(np.sqrt(np.mean(((points[cap_vertices] - c0) @ n) ** 2))), 3),
                            "rim_plane_rms_mm": round(float(np.sqrt(np.mean(((points[rim] - c0) @ n) ** 2))), 3) if len(rim) else None
        }

    checks={"not_closed":qc["boundary_edges"]>0, "several_pieces":qc["components"]!=1,
            "non_manifold": not qc["two_manifold"],"genus_not_0": qc["genus"] != 0, "self_intersections":qc["self_intersecting_faces"]>0,
            "thin_triangles":qc["angles_below_min_pct"]>0.1,"inside_out":qc["volume_ml"]<=0, "no_tv_annulus": qc["tv_annulus_vertices"] == 0
            }

    for name, cap in qc["caps"].items():
        checks[f"{name}_cap_size"]= not CAP_AREA_RANGE[0]<= cap["area_ratio"]<=CAP_AREA_RANGE[1]
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
    note.InsertNextValue("mm, world LPS; chamber 1 RA 2 RV; patch 0 wall 10 SVC inlet 11 IVC inlet 12 RVOT outlet; tv_annulus 0/1")
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

    mask_img = sitk.ReadImage(str(args.case_dir / "mask_rh_cut.nii.gz"))
    openings = json.loads((args.case_dir / "orifices.json").read_text())
    missing = [n for n in OPENINGS if "centre_world" not in openings[n]]
    if missing:                                     # 05 could not build every opening: record it, do not crash
        qc = {"included": False,
              "problems": [f"{n}_opening:{openings[n].get('status', 'missing')}" for n in missing]}
        (args.case_dir / "surface_qc.json").write_text(json.dumps(qc, indent=2))
        print(f"{args.case_dir.name}: included=False {qc['problems']} (05 could not build every opening)")
        raise SystemExit(0)
    
    points, faces = chamber_surface(mask_img)
    points, faces = remesh(points, faces)
    faces = orient_outwards(points, faces)
    face_patch = label_caps(points, faces, openings)
    labels = vertex_labels(points, faces, face_patch, mask_img)
    qc = mesh_qc(points, faces, labels, openings)

    write_vtp(points, faces, labels, args.case_dir / "surface_rh.vtp")
    (args.case_dir / "surface_qc.json").write_text(json.dumps(qc, indent=2))
    print(f"{args.case_dir.name}: included={qc['included']} {qc['problems']}")
    print(f"   {qc['vertices']} vertices, edge {qc['edge_mm_mean']} mm (cv {qc['edge_cv']}), min angle {qc['min_angle_deg']} deg, "
          f"max valence {qc['max_valence']}, volume {qc['volume_ml']} mL, genus {qc['genus']}, "
          f"self-intersections {qc['self_intersecting_faces']}, tv band {qc['tv_annulus_vertices']} vertices")
    for name, cap in qc["caps"].items():
        print(f"   {name}: {cap}")

    