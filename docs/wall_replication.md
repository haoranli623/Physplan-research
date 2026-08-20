# Reproducing the Wall Replication

Run from the repository root on the recorded `torch-gpu` environment. On this Windows machine, set the OpenMP workaround before importing the simulator stack:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$python = 'D:\anaconda\envs\torch-gpu\python.exe'
```

The official checkpoint belongs at `artifacts/checkpoints/jepa_wm_wall.pth.tar`; its SHA256 must be `8EFB0623CFBA1CB3CA210DE26F7579C83DD24936635F11989C515AFCB23BEA1E`.

## Feasibility

```powershell
& $python -m scripts.wall_replication.cache_features
& $python -m scripts.wall_replication.analyze_feasibility
```

## Development and freeze

```powershell
& $python -m scripts.wall_replication.cache_features --config configs/wall_replication/development.yaml
& $python -m scripts.wall_replication.develop_protocol
& $python -m scripts.wall_replication.train_predictor_repair
```

The generated predictor repair is intentionally not committed because it is 70.5 MB. It must match SHA256 `B3C737971D246A779C146364B66D1135A582E7A5F51D9B5FD59F025214398BA9` before independent evaluation.

## Frozen diagnosis

```powershell
& $python -m scripts.wall_replication.cache_features --config configs/wall_replication/diagnosis.yaml
& $python -m scripts.wall_replication.evaluate_diagnosis
```

Do not run repair commands until `wall_bottleneck_prediction.json` is committed. The canonical ordering commits are recorded in `wall_replication_report.md`.

## Repair validation

```powershell
& $python -m scripts.wall_replication.cache_features --config configs/wall_replication/repair.yaml
& $python -m scripts.wall_replication.cache_repaired_predictions --config configs/wall_replication/repair.yaml --output artifacts/wall_replication/repair_repaired_predictions.npz
& $python -m scripts.wall_replication.evaluate_repair
& $python -m scripts.wall_replication.plot_results
```

## Tests

```powershell
& $python -m pytest tests/wall_replication -q
```

Feature caches under `artifacts/wall_replication/` are resumable only at the whole-file level and are ignored by git. Canonical compact results live under `results/wall_replication/`; canonical figures live under `plots/wall_replication/`.
