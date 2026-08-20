# Local Counterfactual Ranking Diagnostic

This experiment follows the inconclusive Boundary-JEPA boundary pilot. It asks
whether true future latents can rank simulator outcomes for a fixed local action
set, and how much ranking utility is lost when those latents are replaced by the
standard JEPA-WM predictions.

## Scientific isolation

- Development data are the old 100-anchor H6 diagnostic cache, because those
  states already influenced project decisions.
- The scorer and all thresholds were frozen in commit `428c2db`.
- The 120-state test split uses seed `24,260,820` and did not exist before the
  freeze commit.
- The scorer reads terminal pooled visual/proprio latents only. It never reads
  raw action, simulator state, contact label, or test cost.
- The scorer is fit once on development GT latents and applied unchanged to GT
  and predicted test latents.
- Every method sees the same 41 saved actions per cloned state.

## Frozen protocol

- Push-T official H6 horizon: 30 controls, five controls per latent step.
- One-sided 41-action angular sweep, magnitude 0.5, half-width 1.2 radians.
- Simulator-only acceptance: exactly one contact/no-contact crossing and
  `max(J_sim) - min(J_sim) >= 0.05`.
- `J_sim = 1 - final_coverage`; lower is better.
- Readout: `StandardScaler + Ridge(alpha=1000)` on terminal global-mean visual
  latent concatenated with terminal proprio latent.
- Alpha was selected with five-fold anchor-grouped GT-only development CV.
- Pair tie tolerance: `1e-4`; top-k: 5.
- Boundary-near adjacent pairs are within two grid cells of the true crossing;
  boundary-far adjacent pairs are at least five cells away and same-mode.
- All confidence intervals bootstrap anchor states, never individual actions.

The complete immutable configuration is
`configs/ranking_diagnostic/protocol.yaml`.

## Reproduction

Use the same environment as the original pilot:

```powershell
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_HOME='D:\projects\physplan\artifacts\torch_cache'
$env:WANDB_MODE='disabled'
$py='D:\anaconda\envs\torch-gpu\python.exe'
```

Reproduce development selection:

```powershell
& $py -m scripts.boundary_jepa.develop_ranking_diagnostic
```

After checking out the freeze commit or later, reproduce fresh generation,
audit, canonical batch-1 feature caching, and evaluation:

```powershell
& $py -m scripts.boundary_jepa.generate_ranking_test
& $py -m scripts.boundary_jepa.audit_labels `
  --sweeps artifacts\ranking_diagnostic\test_sweeps.h5 `
  --output results\ranking_diagnostic\test\label_audit.json `
  --plot plots\ranking_diagnostic\test_label_audit.png `
  --replays 12
& $py -m scripts.boundary_jepa.cache_features `
  --config configs\ranking_diagnostic\protocol.yaml `
  --anchor-batch-size 1
& $py -m scripts.boundary_jepa.evaluate_ranking_diagnostic
& $py -m pytest tests\boundary_jepa -q
```

Generation and feature caching are resumable. Do not rerun test generation with
modified thresholds and call it the same experiment.

## Result

Final decision: **PREDICTOR NOT BOTTLENECK**.

- GT/PRED mean rho: 0.720 / 0.708.
- Delta rho: 0.012, 95% CI [-0.034, 0.058].
- GT/PRED mean regret: 0.0273 / 0.0311.
- Delta regret: 0.0038, CI [-0.0059, 0.0150].
- Boundary-specific excess degradation: -0.055, CI [-0.143, 0.032].

The GT oracle is strong, so task-relevant information is recoverable from the
representation. Predicted latents retain nearly the same ranking quality. The
fixed official terminal goal-L2 score, not predictor fidelity, caused the prior
planner floor under this local protocol.

## Artifacts

- Main report: `ranking_diagnostic_report.md`
- Executive summary: `RANKING_DIAGNOSTIC_SUMMARY.md`
- Frozen model/development results: `results/ranking_diagnostic/development/`
- Fresh raw metrics/scores: `results/ranking_diagnostic/test/`
- Figures: `plots/ranking_diagnostic/`
- Chronological log: `ranking_diagnostic_status.md`
- Raw test trajectories/features: `artifacts/ranking_diagnostic/` (ignored by Git)

The batch-5 engineering check reached approximately 84% of 16 GiB VRAM but
improved throughput by only about 1.2% and introduced small BF16 prediction
differences. Canonical results therefore retain batch=1; the scientific decision
was unchanged under batch=5.

