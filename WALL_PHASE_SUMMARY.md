# Wall Phase Summary

FEASIBILITY RESULT: **PASS** — official model, exact cloning, cost resolution, candidate variation, GT-latent ceiling, throughput, and VRAM checks all passed.

GT-LATENT CEILING: Diagnosis-set rho `0.9428`; normalized regret `0.00359`.

DIAGNOSIS: **PREDICTION** — prediction gap `0.5636` `[0.4871, 0.6373]`; decision-metric gap `-0.0279` with CI crossing zero.

DIAGNOSIS DATASET: 80 anchors, seeds from `1100000`, six diagnosis-only layouts.

PRE-REGISTERED REPAIR: Predictor-only five-epoch standard latent-L2 fine-tune selected on dev latent loss; commit `8559594` precedes repair generation.

REPAIR-VALIDATION DATASET: 80 anchors, seeds from `1200000`, six repair-only layouts; no state, seed, or layout tuple overlaps dev/diagnosis.

REPAIR RESULT: Normalized regret `0.5561 -> 0.1670`; improvement `0.3891` `[0.2930, 0.4804]`; 62 helped, 7 unchanged, 11 harmed.

WRONG-LAYER CONTROL: Baseline-prediction metric readout improvement `0.0069` `[-0.0489, 0.0625]`.

OUT-OF-SAMPLE STATUS: **Fully independent diagnosis → committed repair prediction → unseen repair validation** for Wall.

WHAT REPLICATED: The attribution workflow prospectively selected the higher-value component repair.

WHAT DID NOT REPLICATE: The Push-T bottleneck label. Wall is prediction-limited; Push-T was decision-metric-limited.

PAPER IMPACT: The cross-task contrast is stronger evidence for a diagnostic framework than a second task repeating “L2 is bad.”

NEXT STEP: Write the workshop paper around the two-regime result, then test the frozen attribution logic in the official full-sequence planner as a separate phase.

## What works

- Official Wall model and simulator run reproducibly on the RTX 3080 16 GB.
- Exact cloned-state rollouts and fixed candidate pairing are verified by tests.
- A small action-blind true-latent readout has a high ranking ceiling across disjoint layouts.
- Frozen attribution identifies a prediction bottleneck on Wall.
- The pre-registered predictor repair succeeds on unseen repair states and sharply outperforms the wrong-layer control.

## What failed or remains imperfect

- Predictor repair harms 11/80 repair states.
- Anchor 31 becomes substantially worse despite lower average latent MSE.
- The experiment does not validate official full-sequence CEM.
- Only one predictor-repair training seed was affordable in this phase.

## Commits and ordering

- Initial frozen Push-T result: `ad4a252`.
- Wall protocol freeze: `71e019f`.
- Diagnosis execution preregistration: `960bcd0`.
- Diagnosis plus repair prediction before repair data: `8559594`.
- Final result/report commit: see git log after completion.

## Artifact locations

- Protocol: `configs/wall_replication/protocol.yaml`
- Diagnosis: `results/wall_replication/diagnosis/`
- Repair: `results/wall_replication/repair/`
- Figures: `plots/wall_replication/`
- Predictor repair checkpoint: `results/wall_replication/development/predictor_repair_dev_selected.pth.tar`
- Raw caches: `artifacts/wall_replication/`
- Integrity manifest: `results/wall_replication/artifact_manifest.json`

## Remaining compute

No Wall jobs remain. Reproducing frozen inference caches takes roughly 3–4 minutes per 80-anchor split on this machine. Reproducing the five-epoch predictor development run takes about 19 minutes. Full-sequence Push-T planning was intentionally not started.
