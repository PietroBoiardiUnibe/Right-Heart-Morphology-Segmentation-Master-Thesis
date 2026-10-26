"""
How are the vessel openings found on a REAL heart, and how does each become a footprint on the chamber wall?
One figure per case, one row per opening (SVC, IVC, RVOT), three panels that follow the method:

  1  measured (05)   side view: chamber wall (grey, see-through), the segmented vessel (blue), the vessel axis (red),
                     for the veins the two centreline shells' centres (green: 4-8 and 10-15 mm from the RA, their
                     line IS the axis), the fitted ellipse at the ostium (orange ring) and its centre (black).
                     RVOT: the plane comes from the pulmonary-annulus landmark, the size from the blood pool in it.
  2  footprint (06)  looking down the axis onto the uncut wall: the faces labelled as this vessel's entry zone
                     (orange) and the rim where the score max(rho, tilt / 65 deg) = 1 (black line); rho = distance
                     from the axis / radius, tilt = angle between the wall normal and the axis.
  3  section         the plane through the axis and the ellipse's long axis: wall (black) and vessel (blue)
                     contours, the axis (red), the ellipse as a segment (orange), the window where the vein's cross
                     section was measured (shaded), shell voxels in the plane (green), footprint vertices (orange).

All in the scanner frame (mm). The constants come from 05_cut_orifices.py itself, so the picture follows the method.

Usage (ImageSegmentation/scripts, shapeworks env): python show_openings.py --case <cases>/cas_0019 --out <png>
"""
import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import vtk
from scipy import ndimage as ndi
from skimage.measure import marching_cubes
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

NAMES = {"SVC": (10, 4), "IVC": (11, 5), "RVOT": (12, 3)}      # footprint patch id, vessel label in mask_rh
CROP_MM = 45.0
GREY, BLUE, RED, ORANGE, GREEN, INK = "#9a9a96", "#2a78d6", "#d03a2f", "#eb6834", "#2e9e4f", "#0b0b0b"


def method_constants():
    spec = importlib.util.spec_from_file_location("cut05", Path(__file__).with_name("05_cut_orifices.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def polydata(points, faces):
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly.SetPoints(pts)
    cells = vtk.vtkCellArray()
    cells.SetData(numpy_to_vtk(np.arange(0, 3 * len(faces) + 1, 3, dtype=np.int64), deep=True),
                  numpy_to_vtk(np.ascontiguousarray(faces.ravel(), dtype=np.int64), deep=True))
    poly.SetPolys(cells)
    return poly


def load_surface(case: Path):
    r = vtk.vtkXMLPolyDataReader()
    r.SetFileName(str(case / "surface_chambers.vtp"))
    r.Update()
    p = r.GetOutput()
    P = vtk_to_numpy(p.GetPoints().GetData()).astype(float)
    F = vtk_to_numpy(p.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    return p, P, F, vtk_to_numpy(p.GetCellData().GetArray("patch")), vtk_to_numpy(p.GetPointData().GetArray("patch"))


def crop_mask(img: sitk.Image, arr: np.ndarray, centre: np.ndarray):
    """box of +-CROP_MM around a world point -> (crop (z, y, x), function voxel index (z, y, x) -> world)"""
    sp = np.array(img.GetSpacing())                                    # x, y, z
    idx = np.array(img.TransformPhysicalPointToContinuousIndex([float(v) for v in centre]))   # x, y, z
    half = np.ceil(CROP_MM / sp).astype(int)
    lo = np.maximum(np.floor(idx).astype(int) - half, 0)
    hi = np.minimum(np.floor(idx).astype(int) + half + 1, np.array(arr.shape[::-1]))
    crop = arr[lo[2]:hi[2], lo[1]:hi[1], lo[0]:hi[0]]
    D, o = np.array(img.GetDirection()).reshape(3, 3), np.array(img.GetOrigin())
    to_world = lambda zyx: o + ((np.asarray(zyx)[:, ::-1] + lo) * sp) @ D.T
    return crop, to_world, sp[::-1]


def vessel_mesh(crop, label, to_world):
    if (crop == label).sum() < 50:
        return None
    v, f, _, _ = marching_cubes(np.pad(crop == label, 1).astype(np.float32), 0.5)
    return to_world(v - 1), f


def shells(crop, label, to_world, spacing_zyx, c5):
    """vein voxels in the near / far distance shells from the RA, as 05 uses them (recomputed for display)"""
    d = ndi.distance_transform_edt(crop != 1, sampling=spacing_zyx)
    V = np.argwhere(crop == label)
    dv = d[crop == label]
    near = to_world(V[(dv >= c5.NEAR_SHELL_MM[0]) & (dv <= c5.NEAR_SHELL_MM[1])])
    far = to_world(V[(dv >= c5.FAR_SHELL_MM[0]) & (dv <= c5.FAR_SHELL_MM[1])])
    return near, far


def ring(o):
    t = np.linspace(0, 2 * np.pi, 120)
    c, e1, e2 = (np.array(o[k]) for k in ("centre_world", "axis1_world", "axis2_world"))
    A, B = o["semi_axes_mm"]
    return c + np.outer(A * np.cos(t), e1) + np.outer(B * np.sin(t), e2)


def render(actors, centre, cam_pos, view_up, size=420, parallel=40.0):
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    for a in actors:
        ren.AddActor(a)
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(*centre)
    cam.SetPosition(*cam_pos)
    cam.SetViewUp(*view_up)
    cam.ParallelProjectionOn()
    cam.SetParallelScale(parallel)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.AddRenderer(ren)
    win.SetSize(size, size)
    win.Render()
    grab = vtk.vtkWindowToImageFilter()
    grab.SetInput(win)
    grab.Update()
    img = grab.GetOutput()
    w, h, _ = img.GetDimensions()
    out = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(h, w, -1)[::-1]
    win.Finalize()
    return out


def surface_actor(poly, colour, opacity=1.0, scalars=None, lut=None):
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(poly)
    if scalars is None:
        m.ScalarVisibilityOff()
    else:
        m.SetScalarModeToUseCellFieldData()
        m.SelectColorArray(scalars)
        m.SetLookupTable(lut)
        m.SetScalarRange(0, 1)
        m.ScalarVisibilityOn()
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*matplotlib.colors.to_rgb(colour))
    a.GetProperty().SetOpacity(opacity)
    return a


def line_actor(points, colour, width=3.0):
    poly = vtk.vtkPolyData()
    pts = vtk.vtkPoints()
    pts.SetData(numpy_to_vtk(np.ascontiguousarray(points, dtype=np.float64), deep=True))
    poly.SetPoints(pts)
    lines = vtk.vtkCellArray()
    lines.InsertNextCell(len(points))
    for k in range(len(points)):
        lines.InsertCellPoint(k)
    poly.SetLines(lines)
    m = vtk.vtkPolyDataMapper()
    m.SetInputData(poly)
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*matplotlib.colors.to_rgb(colour))
    a.GetProperty().SetLineWidth(width)
    a.GetProperty().SetRenderLinesAsTubes(True)
    return a


def sphere_actor(centre, colour, radius=1.3):
    s = vtk.vtkSphereSource()
    s.SetCenter(*centre)
    s.SetRadius(radius)
    s.SetThetaResolution(16)
    s.SetPhiResolution(16)
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(s.GetOutputPort())
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(*matplotlib.colors.to_rgb(colour))
    return a


def rim_actor(chamber_poly, score_name):
    poly = vtk.vtkPolyData()
    poly.DeepCopy(chamber_poly)
    poly.GetPointData().SetActiveScalars(score_name)
    cf = vtk.vtkContourFilter()
    cf.SetInputData(poly)
    cf.SetValue(0, 1.0)
    cf.Update()
    m = vtk.vtkPolyDataMapper()
    m.SetInputConnection(cf.GetOutputPort())
    m.ScalarVisibilityOff()
    a = vtk.vtkActor()
    a.SetMapper(m)
    a.GetProperty().SetColor(0, 0, 0)
    a.GetProperty().SetLineWidth(3)
    a.GetProperty().SetRenderLinesAsTubes(True)
    return a


def cut(points, faces, origin, normal):
    """section of a mesh by a plane -> list of 3D polylines"""
    plane = vtk.vtkPlane()
    plane.SetOrigin(*origin)
    plane.SetNormal(*normal)
    c = vtk.vtkCutter()
    c.SetCutFunction(plane)
    c.SetInputData(polydata(points, faces))
    s = vtk.vtkStripper()
    s.SetInputConnection(c.GetOutputPort())
    s.JoinContiguousSegmentsOn()
    s.Update()
    out = s.GetOutput()
    if out.GetNumberOfLines() == 0:
        return []
    P = vtk_to_numpy(out.GetPoints().GetData())
    conn = vtk_to_numpy(out.GetLines().GetConnectivityArray())
    off = vtk_to_numpy(out.GetLines().GetOffsetsArray())
    return [P[conn[off[k]:off[k + 1]]] for k in range(len(off) - 1)]


def show(case: Path, out: Path):
    c5 = method_constants()
    orifices = json.loads((case / "orifices.json").read_text())
    chamber, P, F, face_patch, vert_patch = load_surface(case)
    img = sitk.ReadImage(str(case / "mask_rh.nii.gz"))
    arr = sitk.GetArrayFromImage(img)
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(2)
    lut.SetTableValue(0, *matplotlib.colors.to_rgb("#d9d8d3"), 1.0)
    lut.SetTableValue(1, *matplotlib.colors.to_rgb(ORANGE), 1.0)
    lut.Build()

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for row, (name, (pid, label)) in enumerate(NAMES.items()):
        o = orifices[name]
        c, n = np.array(o["centre_world"]), np.array(o["normal_world"])
        e1, e2 = np.array(o["axis1_world"]), np.array(o["axis2_world"])
        A, B = o["semi_axes_mm"]
        crop, to_world, sp = crop_mask(img, arr, c)
        vessel = vessel_mesh(crop, label, to_world)
        near, far = shells(crop, label, to_world, sp, c5) if name != "RVOT" else (np.zeros((0, 3)), np.zeros((0, 3)))

        # 1  what 05 measured: side view, axis pointing up
        actors = [surface_actor(chamber, "#c9c8c2", 0.35)]
        if vessel is not None:
            actors.append(surface_actor(polydata(*vessel), BLUE, 0.35))
        actors += [line_actor(np.array([c - 15 * n, c + 30 * n]), RED), line_actor(ring(o), ORANGE, 4), sphere_actor(c, INK)]
        for pts in (near, far):
            if len(pts):
                actors.append(sphere_actor(pts.mean(axis=0), GREEN, 1.6))
        ax[row, 0].imshow(render(actors, c + 6 * n, c + 6 * n + 120 * e2, n))
        ax[row, 0].set_title(f"{name}: measured by 05 (d = {o['equivalent_diameter_mm']:.1f} mm, "
                             f"{A:.1f} x {B:.1f} mm)", fontsize=10)

        # 2  footprint on the uncut wall, seen from the vessel
        cham = vtk.vtkPolyData()
        cham.DeepCopy(chamber)
        flag = (face_patch == pid).astype(np.float64)
        arrv = numpy_to_vtk(flag, deep=True)
        arrv.SetName("is_fp")
        cham.GetCellData().AddArray(arrv)
        actors = [surface_actor(cham, "#d9d8d3", 1.0, "is_fp", lut), rim_actor(chamber, f"fp_{name}"),
                  line_actor(ring(o), RED, 2), sphere_actor(c, INK, 1.0)]
        ax[row, 1].imshow(render(actors, c, c + 120 * n, e1, parallel=1.25 * max(A, B) + 8))
        ax[row, 1].set_title(f"{name}: footprint by 06 ({int(flag.sum())} faces), rim = score 1", fontsize=10)

        # 3  section through the axis and the ellipse's long axis
        a3 = ax[row, 2]
        normal = np.cross(e1, n)
        to2 = lambda q: np.column_stack([(q - c) @ e1, (q - c) @ n])
        for poly in cut(P, F, c, normal):
            q = to2(poly)
            a3.plot(q[:, 0], q[:, 1], color=INK, lw=1.4)
        if vessel is not None:
            for poly in cut(vessel[0], vessel[1], c, normal):
                q = to2(poly)
                a3.plot(q[:, 0], q[:, 1], color=BLUE, lw=1.2)
        window = c5.SECTION_WINDOW_MM if name != "RVOT" else (-c5.RVOT_SLAB_MM, c5.RVOT_SLAB_MM)
        a3.axhspan(*window, color=ORANGE, alpha=0.12, label="cross-section measured here")
        for pts, lab in ((near, "near shell"), (far, "far shell")):
            if len(pts):
                inplane = pts[np.abs((pts - c) @ normal) < 1.0]
                q = to2(inplane)
                a3.scatter(q[:, 0], q[:, 1], s=3, color=GREEN, alpha=0.5)
                m2 = to2(pts.mean(axis=0)[None])[0]
                a3.scatter(*m2, s=40, color=GREEN, edgecolor=INK, zorder=5, label=f"{lab} centre")
        fp_pts = P[(vert_patch == pid) & (np.abs((P - c) @ normal) < 1.0)]
        if len(fp_pts):
            q = to2(fp_pts)
            a3.scatter(q[:, 0], q[:, 1], s=10, color=ORANGE, zorder=4, label="footprint vertices")
        a3.plot([-A, A], [0, 0], color=ORANGE, lw=3, label="ostium (ellipse, long axis)")
        a3.axvline(0, color=RED, ls="--", lw=1, label="vessel axis")
        a3.scatter([0], [0], color=INK, s=25, zorder=6)
        a3.set_aspect("equal")
        a3.set_xlim(-CROP_MM * 0.8, CROP_MM * 0.8)
        a3.set_ylim(-CROP_MM * 0.6, CROP_MM * 0.8)
        a3.set_xlabel("along the ellipse's long axis (mm)")
        a3.set_ylabel("along the vessel axis (mm), 0 = ostium")
        a3.set_title(f"{name}: section through the axis", fontsize=10)
        a3.legend(fontsize=7, frameon=False, loc="lower left")
        a3.grid(alpha=0.25)
        for a in ax[row, :2]:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(f"Openings of {case.name}: measured on the vessels (05) -> footprint on the uncut wall (06)",
                 x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", type=Path, required=True, help="case folder: mask_rh.nii.gz, orifices.json, surface_chambers.vtp")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    show(args.case, args.out)