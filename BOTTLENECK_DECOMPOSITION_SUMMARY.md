# Bottleneck Decomposition Summary

## CENTRAL RESULT

**STRONG GO on the frozen local Push-T protocol.** A prospective oracle-substitution diagnosis identified the decision metric as the dominant bottleneck. Replacing only that metric inside the identical 65-query CEM reduced mean simulator regret from **0.1237** to **0.0421**, an improvement of **0.0815** (95% state-bootstrap CI **[0.0563, 0.1079]**). Improving the wrong layer—BF16 to FP32 predictor inference while retaining latent L2—improved regret by only **0.0017** (**[-0.0028, 0.0065]**).

## REPRESENTATION/READOUT CEILING

The frozen action-blind GT-latent ridge readout is useful on 100 fresh anchors: mean Spearman rho **0.696** [0.635, 0.754], pairwise accuracy **0.840**, and reference-set regret **0.0289** [0.0221, 0.0366]. The combined representation/readout gate passes, although it is not a perfect oracle.

## PREDICTION GAP

`G_pred = R_pred_readout - R_GT_readout` is **0.0071** [-0.0051, 0.0198]. It is below the frozen 0.02 material threshold and uncertain around zero.

## DECISION-METRIC GAP

`G_metric = R_pred_L2 - R_pred_readout` is **0.0851** [0.0608, 0.1109]. This is the largest non-overlapping controlled headroom estimate and dominates the prediction and proposal gaps under the frozen rule.

## SEARCH GAP

The best simulator action among all candidates actually queried by baseline CEM is only **0.0074** [0.0042, 0.0115] worse than the 81-action reference oracle. Candidate coverage is not the dominant limitation at this budget.

## SELECTION GAP

Within the candidates queried by L2-CEM, choosing the final L2-CEM action instead of the simulator-best queried action costs **0.1163** [0.0919, 0.1423]. This is the observed downstream manifestation of poor selection and overlaps with the decision-metric gap; the terms are not added.

## DIAGNOSED BOTTLENECK

**DECISION METRIC.** The diagnosis was written to `bottleneck_predictions.json` and committed as `de4d8ca` before either repair result or the FP32 test cache existed.

## TARGETED REPAIR RESULT

Frozen task readout + the same CEM, seeds, reference set, and 65-query budget: mean regret **0.0421** [0.0315, 0.0541]. Improvement over baseline: **0.0815** [0.0563, 0.1079]. It passes the predeclared mean-improvement and CI gates.

## WRONG-LAYER CONTROL RESULT

FP32 predictor + original latent L2 + identical CEM: mean regret **0.1220** [0.0990, 0.1469]. Improvement: **0.0017** [-0.0028, 0.0065]. The targeted mean gain is approximately **48×** larger.

## SECOND-TASK RESULT

Not run. No non-Push-T official checkpoint or validated simulator environment was installed locally. Starting an unvalidated second task after observing Push-T would have weakened, not strengthened, the frozen evidence. Wall is the recommended prospectively configured next task because it has an official public checkpoint, a cloneable low-dimensional simulator, and obstacle-constrained planning complementary to contact-rich Push-T.

## WHAT THE DATA SUPPORTS

On this local H6 Push-T decision problem, the unified controlled substitutions correctly predicted that repairing the decision metric would be high value and predictor numerical precision would be low value. The result supports diagnostic actionability, not merely the familiar observation that latent L2 can be poor.

## WHAT THE DATA DOES NOT SUPPORT

It does not establish a universal bottleneck across tasks, a mathematically additive causal decomposition, superiority on the official full 12-D action-sequence benchmark, or novelty of the ridge scorer itself. The targeted repair harms 17% of individual states and leaves 21% unchanged; aggregate benefit does not eliminate failure cases.

## PUBLICATION POTENTIAL

Strong research-engineering / workshop result and a compelling robotics-simulation portfolio project. A full paper needs one prospectively frozen complementary task and preferably confirmation in the official full-sequence planner rather than only the controlled local finite-grid CEM.

## NEXT STEP

Freeze a Wall decomposition before downloading/evaluating its test split, then test whether the same protocol attributes a different bottleneck. Do not tune the Push-T scorer further.
