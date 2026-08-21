# Boundary-JEPA / Planning Diagnosis Final Summary

## WHAT WORKS

The repository now runs an auditable chain from cloned-state diagnostics through frozen bottleneck decomposition and official-horizon full-sequence CEM evaluation. The local Push-T and Wall experiments show that oracle decomposition can prospectively select the useful repair layer. The official H6 experiment is complete on 30 fresh Push-T episodes with every queried candidate retained.

## WHAT FAILED

The original contact-boundary pilot did not establish that boundary error explains planning regret better than pointwise latent error. More importantly, the local Push-T conclusion does not replicate unchanged under official H6 planning: prediction headroom becomes larger than metric headroom. The matched scorer repair helps on average, but its 95% bootstrap interval includes zero and it harms 5/30 episodes.

## WHAT THE DATA CURRENTLY SUPPORTS

- Local Push-T is metric-limited: prediction gap 0.0071, metric gap 0.0851; the preregistered metric repair improves regret by 0.0815, while the FP32 predictor control improves it by only 0.0017.
- Wall is prediction-limited: prediction gap 0.5636 and metric gap -0.0279; the prediction repair improves regret by 0.3891, while the wrong-layer metric repair changes it by 0.0069.
- Official H6 Push-T changes regime: prediction gap 0.1608 [0.0565, 0.2725], metric gap 0.0950 [0.0026, 0.1978]. Diagnosing the actual planner setting matters.

## WHAT THE DATA DOES NOT SUPPORT

The evidence does not support a universal “JEPA planning is metric-bottlenecked” claim, a general Boundary-JEPA loss claim, or a claim that the H6 scorer repair is statistically established. The paired H6 repair is 0.0843 with CI [-0.00008, 0.1684]. Headroom contrasts are diagnostic, not additive causal components.

## BEST RESULT

The strongest result is the controlled local diagnosis-to-repair loop across two distinct regimes: the decomposition selected metric repair on Push-T and prediction repair on Wall, and each targeted repair substantially beat its preregistered wrong-layer control.

## BIGGEST RISK

Adaptive CEM couples scoring and candidate coverage. In H6 episode 20, the targeted scorer had zero within-trace selection gap but drove search into a worse region, producing regret 1.0 and repair improvement -0.5956. This prevents interpreting fixed-trace scorer quality as guaranteed end-to-end improvement.

## NEXT EXPERIMENT

Pre-register a planner-aware diagnostic that separates prediction, scoring, and adaptive search coverage on fresh episodes, then test whether its state-level diagnosis prospectively routes to predictor, scorer, or search repair. Do not retune the completed Push-T, Wall, or H6 test sets.

## Final H6 protocol and result

- Frozen protocol commit: `3cbd3c50c472f1e3bcaad6e50c3b2835e007094c`.
- Official setup: horizon 6, 30 CEM iterations, 300 candidates, 10 elites, 9,000 model queries per planner.
- Final data: 30 episodes, seeds 20360830–20360859, no filtering, retry, skip, or replacement.
- Baseline regret: 0.4974 [0.3614, 0.6332].
- Targeted-scorer regret: 0.4131 [0.2988, 0.5366].
- Repair counts: helped 16, unchanged 9, harmed 5.
- Frozen decision: **PREDICTION BECOMES IMPORTANT**.

## Reproduction

Use `D:\anaconda\envs\torch-gpu\python.exe` and set `KMP_DUPLICATE_LIB_OK=TRUE` on this machine. Full instructions are in `docs/full_sequence.md`.

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$py='D:\anaconda\envs\torch-gpu\python.exe'
& $py -m scripts.full_sequence.run_final --config configs/full_sequence/protocol.yaml
& $py -m scripts.full_sequence.analyze --config configs/full_sequence/protocol.yaml
& $py -m scripts.full_sequence.plot_results
& $py -m scripts.full_sequence.audit_failure_cases
& $py -m scripts.full_sequence.verify_final_artifacts
```

The final generation command is resumable and skips only already complete paired episode files.

## Artifacts

- Raw H6 episodes: `artifacts/full_sequence/final/episode_000..029.{json,npz}` (~1.3 GB).
- Frozen checkpoint: `artifacts/checkpoints/jepa_wm_pusht.pth.tar`, SHA-256 `9BECA3EAFE0739C3B3ADB5D734FA435CCBDA0FEA8A65D53D4CCCEC176AAAA0EB`.
- Frozen readout: `results/full_sequence/development/frozen_full_sequence_ridge.joblib`, SHA-256 `EDCFED77C89A0625C638FF45A6AC8A8BE9323AE946C5653DAEB6CDC4945EEACC`.
- Machine-readable results: `results/full_sequence/`.
- Figures: `plots/full_sequence/`.
- Current scientific status: `PROJECT_CURRENT_STATUS.md`.
- Workshop draft: `paper/draft.md`.
- Full artifact hashes: `results/full_sequence/artifact_manifest.json`.

## Commits and unfinished work

Key phase commits include `a6bd69f`, `cdebab3`, frozen protocol `3cbd3c5`, integrity guard `6b6ffc3`, serialization-only fix `87b4257`, and paper-first snapshot `33df969`. The final result/report commit is recorded by `git log` after completion.

No research job is intentionally left running. No model retraining is justified by the frozen H6 result without a new preregistered experiment. Re-running all 30 final episodes would take roughly eight GPU-hours on this machine; analysis, plots, and verification take minutes.

## Integrity

Physical labels and task cost remain separate. The action-blind readouts never receive action. Episode initial state is the independent unit. Bad episodes and negative repairs remain in the dataset. The final protocol was committed before final generation, and all final claims derive from saved raw artifacts.

The original 100-anchor Boundary-JEPA pilot remains preserved as an **INCONCLUSIVE / NO-GO for A/B/C/D under that protocol** result. Its report and raw outputs were not redefined to match later findings.
