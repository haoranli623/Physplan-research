# Full-Sequence Push-T Validation Report

## Question

Does the decision-metric bottleneck found in the controlled local Push-T protocol survive official H6 high-dimensional action-sequence planning?

## Status

Development is running. No final episode has been generated or inspected. Result fields below remain explicitly pending until `configs/full_sequence/protocol.yaml` is committed and all final episodes complete.

## Official components retained

- Epoch-50 official Push-T JEPA-WM checkpoint.
- Official image, proprio, and action preprocessing.
- Six model action steps; each contains five sequential normalized 2-D controls.
- 30 diagonal-Gaussian CEM iterations, 300 samples per iteration, 10 elites.
- Mean-sample inclusion, zero initial mean, unit initial variance, and zero momentum.
- Official terminal latent L2 objective with proprio weight 0.1.
- Final action sequence is the updated CEM mean after iteration 30.

The traced implementation has a direct numerical equivalence test against the repository's `CEMPlanner` for sampled actions, CEM losses, and returned mean.

## Necessary local adaptations

The official Push-T offline dataset used for `goal_source: dset` is absent. Initial states are therefore generated from a predeclared, fresh near-contact distribution and evaluated against the canonical simulator target. No episode is filtered using model or simulator outcomes.

The official 300-candidate population caused severe memory pressure when evaluated as one model batch on the 16 GB GPU. The run retains all 300 candidates but scores them in deterministic groups of 75. A full feasibility benchmark measured 302.99 s for 9000 model queries, 270.88 s for simulator replay, and 22.31 s for GT endpoint encoding. This batching adaptation does not change candidates, scores, elites, query count, or CEM updates.

## Development-only readout

Five fresh development episodes generate baseline CEM traces. Simulator endpoints are encoded with the frozen target encoder. One action-blind `StandardScaler + Ridge` readout maps the pooled terminal visual and proprio latent to simulator cost. Alpha is selected from `[0.1, 1, 10, 100, 1000]` by episode-grouped cross-validation MSE. Two development episodes are reused only for targeted-planner integration checks. The selected model and hash will be frozen into the final protocol.

## Final fixed-trace attribution

For each fresh final episode, all 9000 baseline candidates plus the returned final mean are simulator-evaluated. On this identical fixed set, three rankings are computed:

1. encoded simulator endpoint + frozen readout;
2. predicted endpoint + the same frozen readout;
3. predicted endpoint + official latent L2.

Regret is normalized within the baseline trace. The paired episode-level quantities are:

\[
G_{pred}^{full}=R_{pred-readout}-R_{GT-readout},
\]

\[
G_{metric}^{full}=R_{pred-L2}-R_{pred-readout}.
\]

Candidates are not treated as independent observations.

## Matched-budget repair

The targeted run uses the same model, episode, H6 horizon, 30×300 CEM query budget, elite count, initial distribution, and random seed policy. Only the scorer changes from terminal latent L2 to the frozen task readout. Because scores adapt the CEM distribution, baseline and targeted adaptive traces are explicitly not described as matched candidate sets.

For each trace, the simulator-best queried sequence supplies the within-trace search oracle. End-to-end baseline/targeted regret uses the union of their two queried traces as an audit reference, with this limitation stated rather than treating either adaptive trace as an independent global oracle.

## Frozen interpretation rule

The final protocol will fix a material normalized-regret gap and dominance ratio. Outcomes are classified as strong validation, search-dominant regime change, prediction-important regime change, or no clear signal. No scorer, threshold, initial-state distribution, or episode count changes are allowed after final evaluation begins.

## Results

`[FINAL RESULT PENDING]`

## Failure cases

`[FINAL RESULT PENDING]`
