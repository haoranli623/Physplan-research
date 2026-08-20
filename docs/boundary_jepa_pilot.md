# Boundary-JEPA Push-T Feasibility Pilot

This repository branch extends the official Meta JEPA-WMs code at upstream commit
`13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`. The pilot asks whether its frozen
Push-T predictor misplaces local contact/no-contact boundaries despite modest
pointwise latent error.

## Locked protocol

- Each evaluation unit is a cloned seven-dimensional Push-T anchor state.
- Each anchor receives the same predeclared 41-action one-sided angular sweep.
- The primary label is simulator-native any-contact over ten control steps.
- Task coverage/cost is stored separately and never enters the physical label.
- Five 2-D control actions are normalized and concatenated for each official
  JEPA-WM latent step; the pilot predicts two latent steps.
- The action-blind probe sees current latent and GT transition deltas, never action.
- Probe fitting/calibration anchors and the 100 evaluation anchors use disjoint seeds.
- The probe is frozen before it sees predicted features.
- The model and simulator oracle rank the exact same saved candidate actions.

## Environment

The overnight run uses:

```text
D:\anaconda\envs\torch-gpu\python.exe
Python 3.11.15
PyTorch 2.11.0+cu128
RTX 3080 Laptop GPU, 16 GB
```

The official package metadata requests Python 3.10. The exercised Windows Python
3.11 path is therefore a documented compatibility adaptation. Exact simulator replay,
checkpoint loading, preprocessing, and predictor-forward sanity checks must pass
before interpreting results.

Set these process-local variables in PowerShell:

```powershell
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_HOME='D:\projects\physplan\artifacts\torch_cache'
$env:WANDB_MODE='disabled'
$py='D:\anaconda\envs\torch-gpu\python.exe'
```

## Reproduction commands

Download the official public checkpoint to the configured path:

```powershell
New-Item -ItemType Directory -Force -Path artifacts\checkpoints
Invoke-WebRequest `
  -Uri 'https://dl.fbaipublicfiles.com/jepa-wms/pt_jepa-wm.pth.tar' `
  -OutFile 'artifacts\checkpoints\jepa_wm_pusht.pth.tar'
```

Run P0 validation:

```powershell
& $py -m scripts.boundary_jepa.validate_baseline
& $py -m pytest tests\boundary_jepa -q
```

Generate the state-disjoint caches:

```powershell
& $py -m scripts.boundary_jepa.generate_sweeps --split probe_train
& $py -m scripts.boundary_jepa.generate_sweeps --split evaluation
& $py -m scripts.boundary_jepa.audit_labels
```

Cache official frozen encoder/predictor features:

```powershell
& $py -m scripts.boundary_jepa.cache_features `
  --source artifacts\pilot_cache\probe_train_sweeps.h5 `
  --output artifacts\pilot_cache\probe_train_features.h5
& $py -m scripts.boundary_jepa.cache_features `
  --source artifacts\pilot_cache\evaluation_sweeps.h5 `
  --output artifacts\pilot_cache\evaluation_features.h5
```

Freeze the probe and analyze the held-out pilot:

```powershell
& $py -m scripts.boundary_jepa.train_probe
& $py -m scripts.boundary_jepa.analyze_pilot
```

Expensive HDF5/RGB caches and public weights are intentionally ignored by Git but
remain under `artifacts/`. Machine-readable metrics live under `results/pilot/`,
figures under `plots/pilot/`, and the generated scientific report is
`pilot_report.md`.
