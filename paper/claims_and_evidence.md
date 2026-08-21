# Claims and Evidence Ledger

## Headline claim

Different latent world-model planning regimes can expose different recoverable bottlenecks. Candidate-matched oracle substitutions diagnosed intervention-relevant interfaces in local Push-T and Wall, while a separately frozen official H6 validation showed that the diagnosis can change when the planner and candidate family become more realistic.

This is an intervention-oriented empirical claim under three frozen regimes, not a universal or additive causal decomposition.

## Claim P1 — Push-T predictor headroom is small under the frozen local protocol

**Evidence**

- GT/predicted readout regret 0.0289/0.0360.
- Prediction gap 0.0071, 95% CI [-0.0051, 0.0198].
- Same frozen action-blind readout and identical reference actions.

**Limitations**

- Local one-dimensional H6 action slice.
- Predictor failures exist on individual states.

**Unsupported stronger claim**

- JEPA prediction is generally sufficient.
- Improving any predictor cannot help.

## Claim P2 — The dominant Push-T recoverable headroom is downstream in the decision metric

**Evidence**

- Metric gap 0.0851 [0.0608, 0.1109].
- Proposal gap 0.0074; prediction gap 0.0071.
- Latent L2 is weak even on GT futures.
- Frozen dominance rule diagnoses `DECISION METRIC`.

**Limitations**

- Selection and metric quantities overlap and are not additive.
- Task-aligned readout is trained with simulator cost on development states.

**Unsupported stronger claim**

- JEPA planning is generally metric-limited.
- The ridge scorer is algorithmically novel.

## Claim P3 — Push-T diagnosis prospectively predicted a valuable intervention

**Evidence**

- `bottleneck_predictions.json` committed before repair computation.
- Matched 65-query metric repair improves regret by 0.0815 [0.0563, 0.1079].
- FP32 inference + unchanged L2 improves by 0.0017 with CI crossing zero.

**Limitations**

- Diagnosis and repair use the same final states.
- FP32 tests numerical inference precision only.
- Repair harms 17% of states.

**Unsupported stronger claim**

- Fully independent diagnosis-to-repair validation.
- A generally improved predictor was tested and failed.

## Claim W1 — Attribution generalizes to Wall

**Evidence**

- Official model and exact-clone feasibility passed.
- Frozen 80-state Wall diagnosis: prediction gap 0.5636 [0.4871, 0.6373], metric gap -0.0279 with CI spanning zero, proposal gap 0.0333.
- The same dominance rule used in the protocol diagnoses `PREDICTION`, unlike Push-T's `DECISION METRIC`.

**Limitations**

- Controlled 1-D waypoint family, not official continuous CEM.
- Disjoint layout tuples introduce mild distribution shift.

**Unsupported stronger claim**

- Universal validity across world-model architectures, tasks, or planners.

## Claim W2 — Diagnosis predicts repair on unseen Wall states

**Evidence**

- `wall_bottleneck_prediction.json` was committed in `8559594` while the repair cache was absent.
- On 80 subsequently generated repair states, predictor repair improves normalized regret by 0.3891 [0.2930, 0.4804].
- Wrong-layer metric repair improves by 0.0069 [-0.0489, 0.0625].
- Targeted-minus-control contrast: 0.3822 [0.2797, 0.4798].
- Secondary latent MSE falls 0.8564 → 0.1999, consistent with the targeted component changing as intended.

**Limitations**

- Helped/unchanged/harmed: 62/7/11; the 11 harmed states remain visible.
- One repair training seed.
- Predictor fine-tune uses fixed dev data, not a matched offline training-cost comparison to the metric readout.

**Unsupported stronger claim**

- Lower prediction loss guarantees better decisions: anchor 31 is a direct counterexample.
- The framework always finds a single dominant bottleneck.

## Claim X1 — The framework distinguishes regimes rather than always blaming L2

**Evidence**

- Identical conceptual substitutions classify local Push-T as decision metric, Wall as prediction, and official Push-T H6 as prediction important with residual metric headroom.
- Corresponding targeted repairs win over task-appropriate wrong-layer controls in both studies.

**Limitations**

- Task-specific candidate geometries and cost normalizations differ by design.
- Only Wall has a fully disjoint repair-validation set.

**Unsupported stronger claim**

- All failures can be uniquely or additively attributed.
- A two-task, three-regime result establishes broad generality.

## Claim F1 — Official full action-sequence planning

**Status**

- Complete under frozen protocol commit `3cbd3c5`; 30/30 new final episodes, no filtering, retry, or replacement.
- Frozen decision: `PREDICTION BECOMES IMPORTANT`.

**Evidence**

- The fixed baseline trace contains 9000 CEM queries plus returned mean for every state; all three rankings use these identical candidates.
- GT/predicted-readout regrets 0.1724/0.3332; prediction gap 0.1608 [0.0565, 0.2725].
- Predicted-L2 regret 0.4282; metric gap 0.0950 [0.0026, 0.1978].
- Matched-budget adaptive baseline/readout regret 0.4974/0.4131; improvement 0.0843 [-0.00008, 0.1684].
- Helped/unchanged/harmed: 16/9/5.
- Baseline/targeted raw selection gaps 0.1834/0.0983; coverage gaps 0.0297/0.0671.

**Limitations**

- Only 30 states and an explicitly adapted near-contact initial-state generator because the official dataset is unavailable.
- Adaptive traces are not candidate matched after score-dependent CEM updates; their union is only an audit reference.
- The metric-repair CI crosses zero and cannot be called a validated end-to-end improvement.
- No trained predictor repair or wrong-layer comparison has yet tested the H6 diagnosis.

**Unsupported stronger claim**

- Push-T is generally metric limited.
- The local metric repair reliably transfers to iterative high-dimensional planning.
- The state-wise post-hoc metric-gap/repair correlation defines a causal or deployable selector.
