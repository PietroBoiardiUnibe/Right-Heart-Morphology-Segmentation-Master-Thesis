"""
00a_dicom_inventory.py

Read-only inventory of a folder of patient DICOM data, to choose ONE CT series per patient
before anything is anonymised or converted. Only DICOM headers are read (no pixels).

Privacy by design:
  * the printed table and inventory.csv contain NO patient-identifying values: patients are
    numbered p01, p02, ... in folder order, and only technical acquisition fields are kept;
  * the link "patient folder name -> pNN" is written to a SEPARATE file (key_folder_to_pid.csv)
    because folder names are often patient names. Both files go to --out_dir, which must be a
    private, non-synced folder (never the code repo, never a shared OneDrive folder);
  * for every series the script also counts how many identifying DICOM tags are filled in
    (names, IDs, dates, institution ...), without printing their values, so we know what the
    anonymisation step has to remove.

Usage:  python 00a_dicom_inventory.py --root <folder with one subfolder per patient> --out_dir <private folder>
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import pydicom

# files that are certainly not DICOM images (derived meshes, NIfTI, documents): never opened
SKIP_SUFFIXES = (".txt", ".json", ".ini", ".xml", ".pdf", ".stl", ".gz", ".vtk", ".vtp", ".nii", ".csv")

# technical fields kept in the inventory (none of them identifies a person)
FIELDS = ["Modality", "SeriesDescription", "ImageType", "SliceThickness", "PixelSpacing", "Rows", "Columns",
          "KVP", "ConvolutionKernel", "ContrastBolusAgent", "Manufacturer", "ManufacturerModelName",
          "BodyPartExamined", "NominalPercentageOfCardiacPhase", "CardiacSynchronizationTechnique"]

IDENTIFYING = ["PatientName", "PatientID", "PatientBirthDate", "OtherPatientIDs", "PatientAddress",
               "InstitutionName", "InstitutionAddress", "ReferringPhysicianName", "PerformingPhysicianName",
               "OperatorsName", "AccessionNumber", "StudyDate", "SeriesDate", "AcquisitionDate", "StudyID",
               "DeviceSerialNumber", "StationName"]

def read_header(path:Path):
    try:
        return pydicom.dcmread((str(path)), stop_before_pixels=True,force=True)
    except Exception:
        return None

def as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, pydicom.multival.MultiValue)):
        return "\\".join(str(v) for v in value)
    return str(value)

def inventory_patient(folder: Path) -> list[dict]:
    """One row per series found anywhere below `folder`."""
    series = defaultdict(list)
    for f in folder.rglob("*"):
        if f.is_file() and f.suffix.lower() not in SKIP_SUFFIXES:
            ds = read_header(f)
            if ds is not None and "SeriesInstanceUID" in ds:
                series[ds.SeriesInstanceUID].append(ds)
    rows = []
    for uid, headers in series.items():
        first = headers[0]
        z = sorted(float(h.ImagePositionPatient[2]) for h in headers if "ImagePositionPatient" in h)
        steps = [b - a for a, b in zip(z, z[1:])]
        row = {f: as_text(first.get(f)) for f in FIELDS}
        row.update(n_images=len(headers),
                   z_extent_mm=round(z[-1] - z[0], 1) if len(z) > 1 else 0.0,
                   slice_step_mm=round(sorted(steps)[len(steps) // 2], 3) if steps else "",
                   identifying_tags_filled=sum(1 for t in IDENTIFYING if as_text(first.get(t)).strip()),
                   series_key=uid[-8:])                   # tail of the UID, enough to point at a series later
        rows.append(row)
    return sorted(rows, key=lambda r: -r["n_images"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="folder with one subfolder per patient")
    parser.add_argument("--out_dir", type=Path, required=True, help="PRIVATE output folder (not synced, not git)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patients = sorted(p for p in args.root.iterdir() if p.is_dir())
    all_rows, key_rows = [], []
    for i, folder in enumerate(patients, 1):
        pid = f"p{i:02d}"
        key_rows.append({"pid": pid, "folder": folder.name})
        rows = inventory_patient(folder)
        for r in rows:
            all_rows.append({"pid": pid, **r})
        print(f"{pid}: {len(rows)} series")
        for r in rows:
            print(f"   {r['n_images']:5d} img | {r['SeriesDescription'][:40]:40s} | {r['ImageType'][:30]:30s} | "
                  f"thk {r['SliceThickness']:>5s} step {str(r['slice_step_mm']):>6s} | z {r['z_extent_mm']:7.1f} mm | "
                  f"kV {r['KVP']:>4s} | kernel {r['ConvolutionKernel'][:10]:10s} | contrast '{r['ContrastBolusAgent'][:10]}' | "
                  f"phase {r['NominalPercentageOfCardiacPhase']:>4s} | id-tags {r['identifying_tags_filled']}")

    with open(args.out_dir / "inventory.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(args.out_dir / "key_folder_to_pid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pid", "folder"])
        w.writeheader()
        w.writerows(key_rows)
    print(f"{len(patients)} patients, {len(all_rows)} series -> {args.out_dir / 'inventory.csv'}"
          f"\nKEY FILE (identifying, keep private): {args.out_dir / 'key_folder_to_pid.csv'}")