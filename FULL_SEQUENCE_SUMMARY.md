# Full-Sequence Summary

## WHAT WORKS

- The official Push-T H6 action-sequence pipeline runs end to end with the frozen JEPA-WM checkpoint, exact traced CEM updates, 30 iterations, 300 candidates, 10 elites, and 9000 model queries per planner run.
- All 30 precommitted final episodes completed with contiguous seeds 20360830–20360859. No state was filtered, retried, skipped, or replaced.
- Fixed-trace attribution uses the same 9001 retained sequences per state for GT-readout, predicted-readout, and predicted-L2 rankings.
- The matched-budget adaptive comparison changes only the scorer and gives both planners 9000 model queries.

## FROZEN DECISION

`PREDICTION BECOMES IMPORTANT`

The local Push-T decision-metric diagnosis does **not** remain dominant under official high-dimensional H6 sequence planning. Mean normalized prediction headroom is 0.1608 with 95% episode-bootstrap CI [0.0565, 0.2725], while metric headroom is 0.0950 [0.0026, 0.1978]. The frozen prediction-dominance rule is satisfied.

## MATCHED-BUDGET REPAIR

Replacing official terminal latent L2 with the frozen action-blind task readout changes adaptive-CEM regret from 0.4974 [0.3614, 0.6332] to 0.4131 [0.2988, 0.5366]. Mean improvement is 0.0843, but its 95% CI [-0.00008, 0.1684] crosses zero. The state counts are 16 helped, 9 unchanged, and 5 harmed.

This is directional evidence, not a confirmed end-to-end repair. The readout reduces mean within-trace selection gap (0.1834 to 0.0983 raw simulator cost), but its adaptive trace has larger mean union-relative coverage gap (0.0671 versus 0.0297). These separately bootstrapped summaries suggest a selection/coverage tradeoff; they are not a paired causal decomposition.

## STRONGEST COUNTEREXAMPLE

Episode 20 is retained. The targeted scorer chooses the simulator-best sequence in its own trace (selection gap 0), yet that trace misses the useful region found by baseline CEM (union-relative coverage gap 0.5811). Normalized regret rises from 0.4044 to 1.0000, for repair improvement -0.5956. A better scorer can therefore make adaptive search worse by changing which candidates are generated.

## WHAT THE DATA SUPPORTS

- Bottleneck attribution is planner- and candidate-regime dependent.
- Under the official H6 sequence protocol, prediction error is the larger recoverable fixed-trace limitation.
- The simple task readout retains some metric headroom and helps more states than it harms, but does not meet the frozen strong-repair criterion.
- The local Push-T, Wall, and official H6 results together support the principle **diagnose before retraining or rescoring**; they do not support one universal bottleneck.

## WHAT THE DATA DOES NOT SUPPORT

- Push-T or JEPA-WM is generally metric limited.
- The local metric repair reliably transfers to official iterative sequence planning.
- Prediction and metric gaps are additive causal components.
- The adaptive union of two CEM traces is a global action oracle.
- The post-hoc state-level correlation is a deployable gating rule.

## ARTIFACTS

- Frozen protocol: `configs/full_sequence/protocol.yaml`
- Raw episodes: `artifacts/full_sequence/final/episode_000.*` through `episode_029.*`
- Frozen summary: `results/full_sequence/summary.json`
- Episode metrics: `results/full_sequence/episode_metrics.csv`
- Exploratory failure audit: `results/full_sequence/exploratory_failure_audit.json`
- Main plot: `plots/full_sequence/full_sequence_validation.png`
- State-level audit: `plots/full_sequence/statewise_regimes.png`

## NEXT EXPERIMENT

Do not add another task or scorer before submission. The highest-value follow-up is a pre-registered predictor repair under the same official H6 protocol, with a metric repair retained as the wrong-layer control and with an independent/global search reference if computationally feasible.
