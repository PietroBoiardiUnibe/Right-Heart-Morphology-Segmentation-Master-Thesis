# Environment setup

Two environments are realistically needed. Everything below assumes Windows + an NVIDIA
GPU (checked via `nvidia-smi`), consistent with existing IBAMR/ANSYS/COMSOL setup
on this machine.

## 1. Main env: nnUZoo + CCT-FM inference/fine-tuning

nnUZoo owns the dependency set (torch, nnU-Net v2, the ~70 trainer classes). Do **not**
`pip install` CCT-FM's own `requirements.txt` first — nnUZoo pulls everything CCT-FM's
`model_training_and_benchmarks/` scripts need on top of.

```powershell
# from Code/, in VS Code's integrated terminal
cd nnUZoo
uv venv .venv --python 3.12          # or: python -m venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install .                     # or: pip install .
# GPU build (adjust CUDA version to what `nvidia-smi` reports)
uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then add CTAug (only needed once you fine-tune / augment on your own data):

```powershell
cd ..\CTAug
uv pip install .
```

Set the three nnU-Net environment variables (add to your PowerShell profile or a `.env`
you load per session — do **not** hardcode absolute paths in scripts):

```powershell
$env:nnUNet_raw = "C:\Users\boiar\OneDrive - Universitaet Bern\Master Thesis Pietro\Code\ImageSegmentation\data\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\Users\boiar\OneDrive - Universitaet Bern\Master Thesis Pietro\Code\ImageSegmentation\data\nnUNet_preprocessed"
$env:nnUNet_results = "C:\Users\boiar\OneDrive - Universitaet Bern\Master Thesis Pietro\Code\ImageSegmentation\data\nnUNet_results"
```

Sanity check:

```powershell
nnUNetv2_plan_and_preprocess --help
```

## 2. HolOrama (GUI: manual QC, correction, mesh export)

HolOrama has its own `install.ps1`. Run it **without** `-NnUZoo` if you already set up
env 1 above and don't want a second nnUZoo install duplicated inside HolOrama's own venv;
run it **with** `-NnUZoo` if you'd rather HolOrama be fully self-contained and don't mind
the duplicate ~5-10 GB of torch/CUDA packages.

```powershell
cd ..\HolOrama
.\install.ps1                # add -NnUZoo if you want automatic segmentation inside the GUI
```

Known Windows gotchas the install script is supposed to handle automatically (see
HolOrama's own README if it doesn't): a missing `libomp140.x86_64.dll` for torch 2.4,
and an `optree` version conflict (pin to `0.13.1`).

Run it with:

```powershell
python .\src\main.py
```

Check `src/config.yaml` before first use — in particular the `Segmentation` block
(`model_file`, `model_fold`) once you have a checkpoint to point it at.

## Which env for which script

| Script | Env |
|---|---|
| `scripts/00_DICOM_to_NIfTI.py` | env 1 (SimpleITK/pydicom — add if missing: `uv pip install SimpleITK pydicom`) |
| `scripts/01_make_right_heart_dataset_json.py` | env 1 |
| `scripts/02_run_inference_right_heart.py` | env 1 (calls `nnUNetv2_predict` under the hood) |
| `scripts/03_mask_to_chamber_meshes.py` | env 1 (add: `uv pip install vtk trimesh`) |
| HolOrama (manual QC / TV leaflet segmentation / STL export) | env 2 |
