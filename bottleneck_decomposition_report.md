# Where Does Latent Planning Fail?

## 1. Research question

When a latent world-model planner fails, which interface offers the largest recoverable decision-quality headroom: representation/readout, prediction, decision metric, proposal coverage, or within-search selection? The stronger prospective question is whether this diagnosis predicts which intervention will help.

Boundary-JEPA was retired after the previous frozen studies did not support boundary-specific predictor degradation. This project uses that negative result rather than hiding it.

## 2. Starting evidence

The earlier 120-state ranking diagnostic found strong and nearly matched GT/predicted latent utility: rho 0.720/0.708 and regret 0.0273/0.0311. Its prediction regret gap was 0.0038 with a confidence interval crossing zero, while official latent-goal L2 was weak even on true futures. Those states influenced the hypothesis and were not reused for development or final test.

## 3. Environment and model

- Official Meta JEPA-WM Push-T epoch-50 checkpoint.
- Frozen DINOv2 ViT-S/14 target encoder and six-layer AdaLN action-conditioned predictor.
- H6: 30 simulator controls, grouped as six five-control model action chunks.
- Canonical BF16 inference, anchor batch 1; FP32 only for the predeclared wrong-layer control.
- `J_sim = 1 - final_coverage`; lower is better.

## 4. Fresh split discipline

Development used seed 26,260,820 and accepted 80 anchors. The protocol, scorer SHA, dominance rule, test seed, reference grid, CEM budget, and repair gates were frozen in commit `6a49bdd` before final-test data existed.

Final test used seed 28,260,820 and accepted 100 of 191 attempted anchors. There is zero exact state overlap with new development, the earlier 120-state ranking test, or the earlier 100-state pilot. All 100 final anchors are unique.

## 5. Reference candidate set

Each state has the same form of finite local reference set: 81 uniformly spaced actions along a one-sided 1.2-radian angular slice directed away from the block-contact direction, magnitude 0.5, repeated for H6. Acceptance required exactly one simulator contact transition and simulator-cost range at least 0.05. This is a controlled local planning experiment, not the official full action-sequence benchmark.

The finite-set reference oracle is the lowest simulator-cost candidate among those 81 actions.

## 6. Representation/readout substitution

The scorer is `StandardScaler + Ridge(alpha=100)` over terminal global-mean visual latent concatenated with terminal proprio latent. Alpha was selected from five predeclared values by five-fold anchor-grouped GT-only development CV. The scorer never reads action, simulator state, physical label, or final-test cost and is never refit on predictions.

On final GT latents:

- rho 0.696, 95% CI [0.635, 0.754];
- pairwise accuracy 0.840 [0.808, 0.869];
- top-5 retrieval 0.650;
- regret 0.0289 [0.0221, 0.0366].

The combined representation/readout gate passes, while the 0.0289 residual makes clear this is not a perfect representation oracle.

## 7. Prediction substitution

Applying the identical frozen scorer to predicted latents yields rho 0.689 and regret 0.0360. Per-state:

`G_pred = R_pred_readout - R_GT_readout = 0.0071`, CI [-0.0051, 0.0198].

This is below the frozen 0.02 material threshold. State 13 is an important counterexample with `G_pred = 0.181`; the aggregate diagnosis does not claim every state is prediction-safe.

## 8. Decision-metric substitution

On the exact same predicted futures and 81 actions, original latent-goal L2 gives rho 0.162 and regret 0.1212. Replacing only the score with the frozen task readout gives regret 0.0360:

`G_metric = R_pred_L2 - R_pred_readout = 0.0851`, CI [0.0608, 0.1109].

The GT-latent L2 control is also weak (rho 0.023; regret 0.1047), showing that this result is not created by predictor error.

## 9. Search/proposal substitution

The frozen search diagnostic is a scalar finite-grid CEM over the normalized angular coordinate: four iterations, 16 queries/iteration, four elites, plus one explicit final-mean query. Every state therefore has exactly 65 scored queries with identical seed schedules across interventions. Duplicated snapped-grid queries are retained as actual budget consumption; baseline CEM covers a mean 24.4 unique actions.

The simulator-best action among all baseline-queried candidates is only 0.0074 [0.0042, 0.0115] worse than the 81-action reference oracle. Thus the search usually proposes a useful action.

## 10. Selection within search

Baseline selected-action cost minus simulator-best queried cost is 0.1163 [0.0919, 0.1423]. This is not added to `G_metric`: adaptive CEM queries depend on scores, and both quantities expose overlapping consequences of the metric. The report treats every gap as controlled recoverable headroom, not a Shapley value or additive causal attribution.

## 11. Frozen dominance rule and diagnosis

The combined representation/readout gate requires GT rho lower CI above 0.5 and regret upper CI below 0.05. Remaining components require mean gap at least 0.02 with lower CI above zero; a dominant component must be at least 1.5 times the next material component. Selection is a separate fallback class only when metric headroom is not material.

Under this rule the final diagnosis is **DECISION METRIC**. `bottleneck_predictions.json` recorded the evidence, best repair, low-value repair, timestamp, and then-current git commit. Commit `de4d8ca` sealed that prediction before repairs.

## 12. Prospective targeted repair

The targeted intervention replaces latent L2 with the already-frozen task readout inside the exact same 65-query CEM. It changes no predictor weights, proposal family, query budget, search seed, horizon, action grid, or cost.

- Baseline regret: 0.1237 [0.0989, 0.1491].
- Targeted regret: 0.0421 [0.0315, 0.0541].
- Improvement: 0.0815 [0.0563, 0.1079].

The result passes the predeclared improvement threshold of 0.03 and lower-CI-above-zero requirement.

## 13. Wrong-component control

The inexpensive wrong-layer intervention reruns the identical checkpoint and actions in FP32 rather than BF16 but keeps original L2 and the same CEM.

- FP32+L2 regret: 0.1220 [0.0990, 0.1469].
- Improvement: 0.0017 [-0.0028, 0.0065].
- Targeted/wrong mean-improvement ratio: approximately 48.

FP32 changes the frozen readout score by mean absolute 0.00180, but this numerical change does not translate into material simulator regret reduction.

## 14. Failure cases

All categories were selected by deterministic metric rules and saved before reporting:

- State 88: largest GT-readout regret (0.190), showing an imperfect ceiling.
- State 13: largest prediction gap (0.181).
- State 2: latent L2 is surprisingly optimal on the full reference set, while CEM has the largest proposal miss (0.138).
- State 25: targeted repair is harmful by 0.148 because prediction/readout fails locally while L2 happens to work.
- Across all states, targeted repair helps 62%, is unchanged on 21%, and harms 17%.

These cases are retained in raw CSV/NPZ and `failure_case_audit.json`.

## 15. Figures

- `plots/bottleneck_decomposition/system_decomposition.png`
- `plots/bottleneck_decomposition/pusht_bottleneck_profile.png`
- `plots/bottleneck_decomposition/representative_state.png`
- `plots/bottleneck_decomposition/targeted_repair.png`

The representative state is selected deterministically from states with metric gap at least 0.02 and positive repair, closest to that subset's median repair gain.

## 16. Second-task decision

A second task was not run. Only the Push-T task checkpoint was local, and the other candidate environments/checkpoints were not installed or baseline-validated. The clean scientific stopping point is the completed prospective Push-T intervention, rather than an improvised post-result environment port.

Wall is the recommended next prospectively frozen task: it has an official public JEPA-WM checkpoint, cloneable two-dimensional dynamics, and obstacle-constrained proposal geometry complementary to Push-T. Task selection must be committed before its evaluation.

## 17. Scientific interpretation and novelty

Neither learned scoring nor the statement “latent L2 can be bad” is novel. The useful result is the sequence:

1. controlled substitutions quantify component-specific recoverable headroom;
2. a simple frozen rule attributes the bottleneck before repair;
3. a targeted intervention succeeds strongly;
4. a plausible wrong-layer intervention does not.

This is evidence that the diagnostic is actionable on one task. It is not proof of a universal decomposition or causality without interaction.

## 18. Decision and next experiment

Project decision: **STRONG GO on Push-T**.

For a portfolio, the current result is already strong: official model/simulator integration, prospective experiment discipline, state-level uncertainty, negative-result pivot, controlled repair, wrong-layer control, and auditable failures. For publication, the exact next experiment is a frozen Wall replication and then a test in the official full-sequence planner.
