# Boundary-JEPA Overnight Summary

## WHAT WORKS

The official Push-T JEPA-WM checkpoint, simulator cloning, exact counterfactual replay, frozen feature cache, action-blind trajectory probe, boundary extraction, matched-candidate regret, state-level bootstrap, and publication-style plotting all run end to end on this machine. Seven focused tests pass.

## WHAT FAILED

The locked pilot did not establish the proposed decision-critical failure mode. Q2 and Q3 failed. An early float32 cache also failed exact state replay and was quarantined before probe training; the final float64 caches have zero replay failures.

## WHAT THE DATA CURRENTLY SUPPORTS

Contact mode is stably recoverable from held-out true latent transitions: balanced accuracy 0.847, macro-F1 0.847, AUROC 0.933, and ECE 0.017. Transition information matters: the current-state-only control is at chance.

## WHAT THE DATA DOES NOT SUPPORT

On the predeclared 100-state, 41-action local slice, boundary fidelity is not more associated with planning regret than pointwise latent error. The data do not justify Boundary-JEPA intervention training, an L_rel benefit claim, or any A/B/C/D comparison.

## BEST RESULT

The action-blind frozen trajectory probe answers Q1 cleanly while respecting state-disjoint fitting/calibration/evaluation and never reading action.

## BIGGEST RISK

The locked planner diagnostic has a large representation/objective floor: selecting with true future latents has mean simulator regret 0.101 versus 0.108 with predicted latents. The official-horizon H6 falsification gives 0.096 versus 0.108. Consequently, the current regret mostly cannot isolate predictor quality.

## NEXT EXPERIMENT

On fresh development states, prospectively construct a boundary-sensitive local candidate/planning setup and require the true-future-latent selector to rank simulator outcomes sensibly and materially outperform the predicted-latent selector. Freeze that protocol before evaluating new held-out states. Do not tune or relabel the existing 100 evaluation anchors.

## Executive scientific result

Decision: **INCONCLUSIVE**, and operationally **NO-GO for A/B/C/D under the current protocol**.

- Evaluation: 100 cloned held-out anchors × 41 actions = 4,100 rollouts.
- Probe fitting/calibration: 240 disjoint anchors × 41 actions = 9,840 rollouts, split 180/60 by anchor.
- Pointwise latent MSE: mean 0.1459, median 0.1414.
- End-to-end normalized boundary error: mean 0.243, median 0.101.
- GT-probe boundary floor: mean 0.174, median 0.081.
- Planning regret: mean 0.1080, median 0.0560.
- Latent error vs regret: Spearman 0.210, 95% state-bootstrap CI [0.007, 0.390].
- Boundary error vs regret: Spearman 0.089, CI [-0.107, 0.280].
- Difference, boundary minus latent: -0.121, CI [-0.376, 0.116].
- Low-latent/high-boundary state fraction: 0.07.

No action sample was treated as an independent statistical unit, no bad state was dropped, and no primary definition was changed after seeing results.

## Diagnostic result

The decomposition separates simulator boundary, frozen probe on true latents, and frozen probe on predicted latents:

- Only 42/100 GT-probe maps have exactly one crossing; 80/100 predicted-probe maps do; 38/100 are single-crossing in both.
- Median end-to-end boundary error exceeds the GT-probe floor by only 0.014.
- Predictor-versus-GT-probe boundary error has rho 0.142 with H2 regret, CI [-0.066, 0.332].
- A same-anchor, same-action extension to the official six-latent-step horizon retained one simulator boundary in all 100 states and passed 12/12 replay checks.
- At H6, boundary error versus regret remains weak: rho 0.084, CI [-0.124, 0.266].

These diagnostics do not rescue the primary result. They identify two limitations: non-monotone probe probability curves and a planner/representation floor that dominates predictor error.

## Repository state and commits

- Branch: `boundary-jepa-overnight`
- Upstream starting commit: `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`
- `18642e3` — bootstrap official Push-T feasibility pilot
- `875d709` — enforce exact float64 sweep replay
- `56cd9ed` — accelerate endpoint-only sweep rendering
- `5eda1cd` — complete locked feasibility pilot
- Final diagnostics/documentation commit: see `git log -1` after handoff.

## Reproduction

The complete PowerShell sequence and environment notes are in `docs/boundary_jepa_pilot.md`. Core commands after installing dependencies and placing the public checkpoint are:

```powershell
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_HOME='D:\projects\physplan\artifacts\torch_cache'
$env:WANDB_MODE='disabled'
$py='D:\anaconda\envs\torch-gpu\python.exe'

& $py -m scripts.boundary_jepa.validate_baseline
& $py -m pytest tests\boundary_jepa -q
& $py -m scripts.boundary_jepa.generate_sweeps --split probe_train
& $py -m scripts.boundary_jepa.generate_sweeps --split evaluation
& $py -m scripts.boundary_jepa.audit_labels
& $py -m scripts.boundary_jepa.cache_features --source artifacts\pilot_cache\probe_train_sweeps.h5 --output artifacts\pilot_cache\probe_train_features.h5
& $py -m scripts.boundary_jepa.cache_features --source artifacts\pilot_cache\evaluation_sweeps.h5 --output artifacts\pilot_cache\evaluation_features.h5
& $py -m scripts.boundary_jepa.train_probe
& $py -m scripts.boundary_jepa.analyze_pilot
& $py -m scripts.boundary_jepa.diagnose_pilot
```

The H6 commands are documented separately in the reproduction guide. Expensive caches are resumable and ignored by Git.

## Artifacts

- Official checkpoint: `artifacts/checkpoints/jepa_wm_pusht.pth.tar`
- Raw H2 sweep caches: `artifacts/pilot_cache/probe_train_sweeps.h5`, `evaluation_sweeps.h5`
- Frozen H2 features: `artifacts/pilot_cache/probe_train_features.h5`, `evaluation_features.h5`
- H6 diagnostic caches: `artifacts/pilot_cache/evaluation_sweeps_h6_diagnostic.h5`, `evaluation_features_h6_diagnostic.h5`
- Checksums and byte sizes: `results/artifact_manifest.json`
- Primary raw metrics: `results/pilot/state_metrics.csv`, `pilot_summary.json`, `planning_records.npz`
- Diagnostic raw metrics: `results/pilot/state_diagnostics.csv`, `diagnostic_summary.json`, `results/pilot/h6/`
- Frozen probes: `results/pilot/frozen_*_probe.joblib`
- Plots: `plots/pilot/`
- Scientific report: `pilot_report.md`

No Boundary-JEPA training checkpoints exist because the pilot did not pass the intervention gate. The downloaded official epoch-50 checkpoint is the only model checkpoint.

## Unfinished jobs and remaining compute

No process is intentionally left running, and no A/B/C/D job is queued. This is a scientific stop, not a compute failure: using the GPU for intervention training after Q2/Q3 failed would violate the locked protocol.

Estimated compute for the recommended next step, after prospectively fixing the diagnostic on development states:

- 100-state simulator sweep at the current density: about 3 minutes for an H6 same-anchor extension, plus anchor-search overhead if states are resampled.
- Frozen GPU feature caching: about 2.5 minutes for 100×41×6 on this RTX 3080.
- Probe/analysis/plots: under 1 minute once features exist.
- Matched A/B/C/D training: not estimated or authorized until the feasibility gate passes; no positive result should be assumed.

## Integrity notes

- The invalid float32 artifacts are retained under `artifacts/invalid_float32_cache_20260820` with pre-fix audit outputs.
- Final anchors/actions are float64 because contact dynamics amplify truncation at impact.
- Physical contact labels and task costs remain separate.
- Probe training/calibration/evaluation splits are disjoint by anchor seed.
- All model/oracle comparisons use the exact same saved action sets.
- The primary pilot remains INCONCLUSIVE after diagnostics; H6 is labeled secondary.
