# Related Work Notes

These are routing notes, not final prose. Verify every citation against primary papers before writing.

## Latent world models for physical planning

- JEPA-WMs: official repository/paper underlying the checkpoints and planning baselines used here. Position this work as diagnosis of planning interfaces, not a new representation or predictor architecture.
- DINO-WM: visual world-model planning baseline; relevant to representation and latent-distance objectives.
- V-JEPA 2 action-conditioned models: relevant action-conditioned latent prediction context.

## Learned planning metrics

- Temporal-distance, reachability, goal-conditioned value, and contrastive planning objectives already establish that Euclidean latent distance can be misaligned.
- Our claim must not be “learned metric beats L2.” Distinction: controlled component attribution selects the repair before intervention.

## Model accuracy versus control utility

- Prior model-based RL and world-model work separates predictive fidelity from downstream return.
- Distinction to establish: the present protocol estimates recoverable headroom at several interfaces using identical candidate sets and simulator oracles.

## Search/proposal diagnosis

- CEM/MPC performance depends on candidate coverage and within-set selection.
- Our audit retains every queried candidate and separately measures simulator-best queried cost versus the finite reference oracle.

## Diagnostic methodology

- Look for primary work on oracle substitutions, component-wise upper bounds, and prospective intervention validation.
- Avoid “first” claims. Novelty rests on the full diagnosis → committed repair prediction → independent intervention sequence.

## Citation verification queue

- Confirm exact JEPA-WMs, DINO-WM, and V-JEPA 2 bibliographic entries from official papers.
- Select 2–3 primary temporal/reachability metric papers.
- Select 1–2 primary model-based control papers explicitly studying prediction/control mismatch.
- Select one primary CEM/MPC reference for proposal/selection terminology.
