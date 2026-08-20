# Reproducing the Push-T Bottleneck Decomposition

## Scientific chronology

1. New development states were generated and used for readout/CEM choices.
2. Commit `6a49bdd` froze `configs/bottleneck_decomposition/protocol.yaml` before final-test data existed.
3. One frozen final decomposition diagnosed `DECISION METRIC`.
4. Commit `de4d8ca` saved `bottleneck_predictions.json` before repairs.
5. The matched readout repair and FP32-predictor wrong-layer control were run afterward.

Do not regenerate a test split with changed thresholds and call it the same experiment.

## Environment

```powershell
$env:SDL_VIDEODRIVER='dummy'
$env:PYGAME_HIDE_SUPPORT_PROMPT='1'
$env:TORCH_HOME='D:\projects\physplan\artifacts\torch_cache'
$env:WANDB_MODE='disabled'
$py='D:\anaconda\envs\torch-gpu\python.exe'
```

Place the official public checkpoint at `artifacts/checkpoints/jepa_wm_pusht.pth.tar`.

## Development reproduction

```powershell
& $py -m scripts.boundary_jepa.generate_bottleneck_sweeps `
  --config configs\bottleneck_decomposition\development.yaml

& $py -m scripts.boundary_jepa.audit_labels `
  --sweeps artifacts\bottleneck_decomposition\development_sweeps.h5 `
  --output results\bottleneck_decomposition\development\label_audit.json `
  --plot plots\bottleneck_decomposition\development_label_audit.png `
  --replays 12

& $py -m scripts.boundary_jepa.cache_features `
  --config configs\bottleneck_decomposition\development.yaml `
  --source artifacts\bottleneck_decomposition\development_sweeps.h5 `
  --output artifacts\bottleneck_decomposition\development_features.h5 `
  --anchor-batch-size 1

& $py -m scripts.boundary_jepa.develop_bottleneck_decomposition
```

## Frozen final decomposition

```powershell
& $py -m scripts.boundary_jepa.generate_bottleneck_sweeps `
  --config configs\bottleneck_decomposition\protocol.yaml

& $py -m scripts.boundary_jepa.audit_labels `
  --sweeps artifacts\bottleneck_decomposition\test_sweeps.h5 `
  --output results\bottleneck_decomposition\test\label_audit.json `
  --plot plots\bottleneck_decomposition\test_label_audit.png `
  --replays 12

& $py -m scripts.boundary_jepa.cache_features `
  --config configs\bottleneck_decomposition\protocol.yaml `
  --source artifacts\bottleneck_decomposition\test_sweeps.h5 `
  --output artifacts\bottleneck_decomposition\test_features_bf16.h5 `
  --anchor-batch-size 1

& $py -m scripts.boundary_jepa.evaluate_bottleneck_decomposition
```

Commit `bottleneck_predictions.json` before continuing.

## Prospective repairs

```powershell
& $py -m scripts.boundary_jepa.evaluate_bottleneck_repairs --stage targeted

& $py -m scripts.boundary_jepa.cache_features `
  --config configs\bottleneck_decomposition\protocol.yaml `
  --source artifacts\bottleneck_decomposition\test_sweeps.h5 `
  --output artifacts\bottleneck_decomposition\test_features_fp32.h5 `
  --no-bfloat16 --anchor-batch-size 1

& $py -m scripts.boundary_jepa.evaluate_bottleneck_repairs --stage final
& $py -m scripts.boundary_jepa.plot_bottleneck_decomposition
& $py -m pytest tests\boundary_jepa -q
```

## Outputs

- Frozen config: `configs/bottleneck_decomposition/protocol.yaml`
- Development artifacts/scorer: `results/bottleneck_decomposition/development/`
- Test decomposition: `results/bottleneck_decomposition/test/`
- Prospective repair/control: `results/bottleneck_decomposition/repair/`
- Raw expensive caches: `artifacts/bottleneck_decomposition/` (git-ignored)
- Figures: `plots/bottleneck_decomposition/`
- Executive summary: `BOTTLENECK_DECOMPOSITION_SUMMARY.md`
- Full report: `bottleneck_decomposition_report.md`

The finite-grid local CEM has 65 actual score queries but fewer unique snapped actions. Raw query indices and iterations are saved, so proposal coverage and selected actions can be audited exactly.
