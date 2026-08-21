# Full-Sequence Push-T Validation Report

## Question

Does the decision-metric bottleneck found in the controlled local Push-T protocol survive official H6 high-dimensional action-sequence planning?

## Answer

No, not as the dominant aggregate bottleneck. The frozen final rule returns `PREDICTION BECOMES IMPORTANT`. Prediction headroom is 0.1608 [0.0565, 0.2725], larger than metric headroom 0.0950 [0.0026, 0.1978]. A matched-budget metric repair improves mean regret by 0.0843, but its interval [-0.00008, 0.1684] crosses zero.

## Protocol integrity

- Protocol commit before final generation: `3cbd3c50c472f1e3bcaad6e50c3b2835e007094c`.
- Thirty fresh final states, seeds 20360830–20360859.
- No outcome filtering, retries, skips, replacements, or discarded states.
- Independent unit: episode initial state.
- Nonparametric episode bootstrap: 5000 samples, seed 20260901.
- Frozen material gap 0.02 and dominance ratio 1.5.
- Raw candidates, scores, CEM distributions, simulator outcomes, and endpoint features are retained.

## Official components retained

- Epoch-50 official Push-T JEPA-WM checkpoint, SHA-256 `9BECA3EAFE0739C3B3ADB5D734FA435CCBDA0FEA8A65D53D4CCCEC176AAAA0EB`.
- Official preprocessing, action normalization, H6 rollout, and terminal latent-L2 score with proprio weight 0.1.
- Six model action steps, each containing five normalized 2-D controls: 60 scalar action dimensions and 30 executed controls.
- Thirty diagonal-Gaussian CEM iterations, 300 candidates per iteration, 10 elites, mean-sample inclusion, zero initial mean, unit variance, zero momentum, and returned final mean.
- Exactly 9000 model queries for each baseline and targeted planner.

The traced CEM implementation passed a direct numerical equivalence test against the repository `CEMPlanner` for sampled actions, losses, distribution updates, and returned mean.

## Necessary local adaptations

The official offline Push-T dataset requested by `goal_source: dset` is unavailable locally. States are generated from the predeclared `fresh_actionable_near_contact_v1` distribution and use the canonical target. This limits direct comparability to an official dataset-sourced evaluation.

A literal 300-candidate FP32 model batch exceeded practical laptop memory. Each unchanged 300-candidate population is scored in four deterministic groups of 75. Samples, scores, elite selection, updates, and query count are unchanged. A full benchmark measured 302.99 seconds for 9000 model queries, 270.88 seconds for simulator replay, and 22.31 seconds for GT endpoint encoding.

## Frozen readout

The task scorer is `StandardScaler + Ridge(alpha=0.1)`, trained only on GT endpoint latents from five development episodes and selected by development-episode-grouped cross-validation. It reads pooled terminal visual and proprio features, never action. Its artifact SHA-256 is `EDCFED77C89A0625C638FF45A6AC8A8BE9323AE946C5653DAEB6CDC4945EEACC`; Torch and sklearn outputs agree within 2.38e-7.

## Fixed-trace attribution

Every baseline CEM query plus the returned mean is simulator-evaluated. The identical 9001 sequences per episode are ranked by:

1. GT endpoint latent + frozen task readout;
2. predicted endpoint latent + the same readout;
3. predicted endpoint latent + official terminal L2.

| Quantity | Mean | Median | 95% episode-bootstrap CI |
|---|---:|---:|---:|
| GT-readout regret | 0.1724 | 0.1216 | [0.1065, 0.2529] |
| Predicted-readout regret | 0.3332 | 0.2307 | [0.2226, 0.4601] |
| Predicted-L2 regret | 0.4282 | 0.4009 | [0.2945, 0.5634] |
| Prediction gap | **0.1608** | 0.0340 | **[0.0565, 0.2725]** |
| Metric gap | 0.0950 | 0.0000 | [0.0026, 0.1978] |

Prediction gap clears the 0.02 material threshold and its mean exceeds 1.5 times the metric-gap mean. Therefore the frozen classification is `PREDICTION BECOMES IMPORTANT`.

The positive metric-gap interval still indicates residual decision-metric headroom on this fixed trace. The result is a dominance statement, not a claim that the metric is irrelevant.

## Matched-budget adaptive repair

Baseline and targeted CEM use the same model, initial state, horizon, iteration/sample/elite counts, initial distribution, and query budget. Only the score changes. Because scores update the proposal distribution, the two adaptive traces are not candidate matched. Their union is an audit reference, not a global oracle.

| Quantity | Mean | Median | 95% episode-bootstrap CI |
|---|---:|---:|---:|
| Baseline normalized regret | 0.4974 | 0.5344 | [0.3614, 0.6332] |
| Targeted normalized regret | 0.4131 | 0.3783 | [0.2988, 0.5366] |
| Repair improvement | 0.0843 | 0.0233 | **[-0.00008, 0.1684]** |
| Baseline selection gap, raw | 0.1834 | 0.1342 | [0.1232, 0.2454] |
| Targeted selection gap, raw | 0.0983 | 0.0792 | [0.0636, 0.1363] |
| Baseline coverage gap, raw | 0.0297 | 0.0000 | [0.0090, 0.0549] |
| Targeted coverage gap, raw | 0.0671 | 0.0017 | [0.0295, 0.1168] |

Helped/unchanged/harmed counts are 16/9/5. The repair direction is favorable on average, but the frozen strong-validation condition requires a strictly positive lower confidence bound and is not met.

## Failure cases and mechanism boundary

- **Episode 20 (strongest harm):** baseline/targeted regret 0.4044/1.0000; improvement -0.5956. The targeted within-trace selection gap is exactly zero, but its union-relative coverage gap is 0.5811. The scorer selects the best candidate it generated while steering CEM into a poor candidate region.
- **Episode 17:** prediction gap 0.3543 and metric gap -0.3522; repair worsens regret by 0.3804. Both targeted selection and coverage are worse.
- **Episodes 1 and 27:** metric gaps 0.7084 and 0.9925; repair improvements 0.5856 and 0.4158. These retained states show that the metric repair remains useful in metric-dominant regimes.

A clearly labeled post-hoc audit finds Spearman correlation 0.792 [0.493, 0.955] between per-state metric gap and repair improvement. This is descriptive, shares quantities with the repair evaluation, and cannot be treated as causal evidence or a learned gating policy.

## Interpretation

The realistic planner changes the diagnosis. Local nearby-action slices suppress accumulated prediction and adaptive proposal effects; full 60-D sequence search exposes them. The metric readout improves ranking on many states but also changes where CEM searches, creating a selection/coverage interaction absent from candidate-matched attribution.

The supported conclusion is therefore:

> Under official Push-T H6 sequence planning, prediction is the dominant recoverable fixed-trace bottleneck. Metric headroom remains, but the precommitted metric repair has uncertain end-to-end benefit because adaptive search can lose coverage.

## Reproduction

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.run_final --config configs/full_sequence/protocol.yaml
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.analyze --config configs/full_sequence/protocol.yaml
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.plot_results
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.audit_failure_cases
```

The final runner is resumable: existing completed episode JSON files are skipped. The analysis refuses to run unless all 30 frozen JSON/NPZ pairs and expected episode/seed identities are present.
