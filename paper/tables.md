# Paper Tables

All values below are copied from frozen machine-readable artifacts. Confidence intervals are 95% state-level bootstrap intervals. Lower normalized regret is better.

## Table 1 — Controlled bottleneck diagnosis

| Task / protocol | States | GT-readout regret | Pred-readout regret | Prediction gap | Metric gap | Proposal gap | Diagnosis |
|---|---:|---:|---:|---:|---:|---:|---|
| Push-T, local H6 angular family | 100 | 0.0289 | 0.0360 | 0.0071 [-0.0051, 0.0198] | **0.0851 [0.0608, 0.1109]** | 0.0074 | Decision metric |
| Wall, local waypoint family | 80 | — | — | **0.5636 [0.4871, 0.6373]** | -0.0279 (CI spans zero) | 0.0333 | Prediction |

Notes: gaps are paired contrasts on identical state-specific candidate sets and are not additive. Wall's GT/pred readout regrets should be inserted from the machine-readable summary before camera-ready formatting rather than reconstructed from rounded gaps.

## Table 2 — Pre-specified repair validation

| Task | Validation states | Baseline regret | Targeted regret | Targeted improvement | Wrong-layer improvement | Targeted vs control |
|---|---:|---:|---:|---:|---:|---:|
| Push-T | 100 shared final states | 0.1237 | 0.0421 | **0.0815 [0.0563, 0.1079]** | 0.0017 (CI spans zero) | — |
| Wall | 80 disjoint repair states | 0.5561 | 0.1670 | **0.3891 [0.2930, 0.4804]** | 0.0069 [-0.0489, 0.0625] | **0.3822 [0.2797, 0.4798]** |

Notes: Push-T's control is FP32 predictor inference with the unchanged latent-L2 scorer; it is not a trained predictor improvement. Wall's targeted repair is a predictor repair and its control is a metric readout.

## Table 3 — Mechanism and heterogeneity checks

| Task | Check | Result | Interpretation boundary |
|---|---|---:|---|
| Push-T | GT vs predicted readout Spearman | 0.696 vs 0.689 | Similar average decision ordering under the frozen readout; not latent equivalence |
| Wall | Prediction MSE, baseline → repair | 0.8564 → 0.1999 | Targeted component changed as intended; not sufficient for every decision |
| Wall | Helped / unchanged / harmed states | 62 / 7 / 11 | Aggregate gain is heterogeneous; no silent seed/state removal |
| Wall | Retained failure, anchor 31 regret | 0.215 → 0.991 | Lower MSE does not guarantee lower planning regret |

## Table 4 — Full action-sequence placeholder

Do not populate until `configs/full_sequence/protocol.yaml` has been committed and final evaluation has completed.

| Protocol | Episodes | Queries/episode | Prediction gap | Metric gap | Targeted repair | Wrong-layer control | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Official Push-T H6 CEM | pending | pending | pending | pending | pending | pending | pending |
