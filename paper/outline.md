# Where Does Latent Planning Fail?

## Working subtitle

Controlled Bottleneck Attribution for World-Model Planning

## Abstract placeholder

- Problem: planning failure is often attributed to world-model prediction without isolating downstream interfaces.
- Method: controlled oracle substitutions estimate representation/readout, prediction, decision-metric, proposal, and within-search headroom.
- Push-T result: prediction headroom is small; metric headroom is large; diagnosis was committed before a matched repair.
- Push-T intervention: metric repair substantially outperforms an inference-precision control.
- Wall independent diagnosis/repair result: prediction gap 0.564; predictor repair improves normalized regret by 0.389 on unseen states, while the wrong-layer metric repair is near zero.
- Full action-sequence validation: **future placeholder; not part of the current Wall phase**.

## 1. Introduction

- Latent world-model planners fail through several coupled interfaces.
- Prediction loss alone does not identify the component worth repairing.
- Principle: diagnose before retraining the world model.
- Conservative current claim is limited to the frozen local Push-T protocol.
- Contributions should be framed as attribution protocol, prospective repair selection, and—if Wall succeeds—independent validation.

## 2. Controlled Bottleneck Attribution

### 2.1 Planning pipeline

Representation/readout → prediction → decision metric → search/proposal → selected action → simulator outcome.

### 2.2 Controlled substitutions

- GT representation + frozen task readout: combined representation/readout ceiling.
- GT → predicted latent under identical readout: prediction headroom.
- Default objective → task readout on identical predicted futures: metric headroom.
- Simulator oracle over actual queried candidates → finite reference oracle: proposal coverage.
- Selected queried action → simulator-best queried action: within-search selection.

### 2.3 Statistical and causal boundary

- Anchor state is the independent unit.
- State bootstrap and effect sizes, not action-level pseudo-replication.
- Gaps need not add; interactions are explicit.
- Attribution predicts a repair but is not automatically a causal decomposition.

## 3. Push-T Local Planning Case Study

### 3.1 Frozen protocol

100 fresh anchors, 81 reference actions, H6, action-blind ridge, finite-grid 65-query CEM.

### 3.2 Diagnosis

- GT readout rho 0.696; regret 0.0289.
- Prediction gap 0.0071, CI crosses zero.
- Metric gap 0.0851, positive CI.
- Proposal gap 0.0074.
- Frozen diagnosis: decision metric.

### 3.3 Pre-specified intervention

- Diagnosis commit preceded repair computation.
- Metric readout repair: regret 0.1237 → 0.0421.
- Improvement 0.0815 [0.0563, 0.1079].
- FP32 inference + L2 control improvement 0.0017, CI crosses zero.
- Phrase control narrowly: increasing predictor inference precision did not recover performance.

### 3.4 Limitation

Diagnosis and repair share the same final state set, despite correct temporal preregistration. This is not independent out-of-sample repair validation.

## 4. Independent Wall Replication

- Official model/environment feasibility passed, including exact clone replay.
- Three-way split: 60 dev / 80 diagnosis / 80 repair, with disjoint seeds, states, and layout tuples.
- Protocol commit `71e019f`; repair prediction commit `8559594` precedes repair generation.
- Diagnosis: prediction gap 0.5636 [0.4871, 0.6373]; metric gap -0.0279 with CI crossing zero; frozen label `PREDICTION`.
- Independent repair: normalized regret 0.5561 → 0.1670; improvement 0.3891 [0.2930, 0.4804].
- Wrong-layer metric readout: improvement 0.0069 [-0.0489, 0.0625].
- Cross-task story: the same framework selects different components rather than encoding a fixed anti-L2 conclusion.

## 5. Full-Sequence Planning

Placeholder only. Do not add experiments during the Wall phase.

## 6. Related Work

- Latent world-model planning and action-conditioned prediction.
- Goal/reachability/temporal metric learning for planning.
- Search and proposal bottlenecks in model-predictive control.
- Diagnostic and oracle-substitution evaluation.

## 7. Limitations

- Two controlled tasks, but only one independently separated repair set.
- Controlled local candidate geometry rather than official full sequence search.
- Readout ceiling combines representation and readout capacity.
- Headroom interactions prevent additive causal attribution.
- Precision control is not generic predictor improvement.

## 8. Conclusion

Diagnose the limiting interface before spending compute on the wrong component. Push-T and Wall expose different regimes, and the committed Wall diagnosis selected the effective repair on unseen states.
