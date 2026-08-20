# Claims and Evidence Ledger

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

- Pending feasibility, frozen diagnosis, and unseen repair validation.

**Limitations**

- No claim until all three Wall splits and temporal commits exist.

**Unsupported stronger claim**

- Cross-task generality based only on Push-T.

## Claim W2 — Diagnosis predicts repair on unseen Wall states

**Evidence**

- Pending `wall_bottleneck_prediction.json` commit and untouched repair set.

**Limitations**

- Must report helped/unchanged/harmed fractions and null results.

**Unsupported stronger claim**

- Independent validation before repair-set evaluation is complete.
