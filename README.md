# Right Heart Automated Segmentation Pipeline

This folder is Pietro's own orchestration layer for the "get patient-specific right
heart chamber meshes ready for SSM" milestone of the BPVT/TTVR thesis. It does not
duplicate any of the external tools' code — it sits on top of them, as siblings under
`Code/`:

```
Code/
├── CCT-FM/            <- model training/inference/eval code (arXiv:2607.11287)
├── nnUZoo/             <- nnU-Net-based model zoo: CNN/Transformer/Mamba trainers, incl. the
│                           pretrained 14-structure cardiac CT model CCT-FM is built on
├── CTAug/              <- cardiac-CT-specific augmentation library (metal/wire/calcification/
│                           step-and-shoot), used during CCT-FM training, reusable standalone
├── HolOrama/            <- desktop GUI: manual/semi-automated CCTA segmentation, 3D render,
│                           brush/lasso correction tools, NIfTI mask -> STL mesh export
├── Licenses&Credits/    <- license notices + consolidated citation list for every tool below
└── ImageSegmentation/   <- YOU ARE HERE: pipeline glue, dataset.json templates, scripts, notes
```

## Where this fits in the thesis pipeline

```
[patient CCT DICOM] -> [00: DICOM->NIfTI + resample/normalize]
                     -> [01: right-heart dataset.json + label re-mapping]
                     -> [02: inference with CCT-FM/nnUZoo pretrained weights]
                            -> RA + RV + (partial annulus) label map, 14-structure model
                     -> [manual QC / correction in HolOrama, esp. TV leaflets + annulus detail]
                     -> [03: NIfTI multi-label mask -> per-chamber watertight surface mesh]
                     -> >>> feeds into the SSM stage (ShapeWorks / Deformetrica) <<<
```

This is the stage the project notes call "the gating milestone before FSI" — nothing
here should be considered final until segmentation is checked structure-by-structure
against source imaging.

## What each tool actually gives you here

- **CCT-FM** — the paper's own training/inference/benchmark code and the *reference
  document* for which trainer name, plans and pretrained checkpoint to use. You will
  mostly *read* this repo (esp. `model_training_and_benchmarks/README.md`) rather than
  run it directly for your own data — its `train.py` is written to reproduce the paper's
  benchmark grid, not as a generic "segment my patient" entry point.
- **nnUZoo** — the actual engine. This is what you `pip install` and call via
  `nnUNetv2_predict` / `nnUNetv2_train`. It owns the pretrained 14-structure cardiac CT
  model's architecture and trainer classes. **This is the one true dependency for
  inference on your own scans.**
- **CTAug** — only relevant if/when you fine-tune on your own annotated cohort and want
  robustness to metal artifacts (relevant given TTVR devices are metallic partly in the nitinol frames) or gated-acquisition step artifacts. Not needed for a first pass of pure inference with
  frozen pretrained weights.
- **HolOrama** —  manual QC and correction layer, and your bridge from NIfTI masks to
  STL meshes. Also where  will hand-segment the tricuspid valve leaflets and annulus,
  since the pretrained 14-structure model does **not** include them (explicitly flagged
  as a limitation by the CCT-FM authors).

## Environment setup

See [`environment/ENVIRONMENT.md`](environment/ENVIRONMENT.md).

## Pipeline scripts

See [`scripts/`](scripts). These are stubs/templates you fill in with your actual data
paths and cohort-specific choices — they are not meant to run as-is.

## Data layout

`data/` mirrors the nnU-Net v2 raw/preprocessed/results layout nnUZoo expects. It is
git-ignored (imaging data does not belong in a git repo) — see `.gitignore`.

## What this pipeline does *not* yet cover 

1. **Valve leaflets, chordae, precise annular geometry** — not part of the pretrained
   14-structure model. Plan on manual/semi-automated delineation in HolOrama (or ITK-SNAP)
   on top of the automated RA/RV backbone, exactly as already written into the thesis
   SSM section. Manually embed them afterwards on computational domains.
2. **Mask -> SSM-ready mesh is not automatic.** nnUZoo/CCT-FM output NIfTI label maps.
   HolOrama can export STL from its own brush/lasso-edited masks, but you still need a
   deliberate step to get a *clean, watertight, per-chamber* surface (remove islands,
   close small gaps, smooth without eroding volume) before SSM/CGAL meshing — this is
   `03_mask_to_chamber_meshes.py`
3. **License note (read `Licenses&Credits/CCT-FM_LICENSE_NOTICE.md`):** CCT-FM's own
   repository (code + weights + dataset) is released under **CC-BY-NC-ND 4.0**, not
   Apache/MIT like the other three. Academic non-commercial use with attribution is
   explicitly permitted, but redistributing modified weights/derivatives is not.
