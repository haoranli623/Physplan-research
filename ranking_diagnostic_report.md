# Local Counterfactual Ranking Diagnostic

## 1. Why the original pilot was inconclusive

The original fixed latent-goal planner produced almost the same regret from true and predicted futures, so its planner/representation floor obscured predictor quality.

## 2. New hypothesis

True future latents may expose local action utility even when the fixed latent-goal distance does not; the matched predicted-latent score measures information lost by prediction.

## 3. Development/test split

The old 100 H6 anchors were development-only. The protocol and scorer were frozen in commit `428c2db` before generating 120 new seed-disjoint test anchors.

## 4. Candidate-set construction

Each anchor uses 41 identical one-sided angular candidates, repeated for H6/30 controls. A simulator-only frozen rule requires exactly one contact boundary and cost range at least 0.05.

## 5. Simulator cost definition

`J_sim = 1 - final_coverage`; lower is better. Raw trajectories, coverage, contact steps, actions, and states remain in the resumable HDF5 cache.

## 6. GT latent scoring protocol

StandardScaler + Ridge(alpha=1000.0) consumes only terminal global-mean visual latent plus terminal proprio latent. It was selected with five-fold anchor-grouped development CV and fit on development GT latents only. It never reads action.

## 7. Predicted latent scoring protocol

The identical frozen scorer is applied to standard JEPA-WM predicted terminal latents. It is not refit, recalibrated, or tuned for predictions.

## 8. Metrics

Per-state Spearman, pairwise accuracy above 0.0001 cost tolerance, top-5 retrieval, selection regret, delta-rho, and delta-regret. Anchor is the statistical unit.

## 9. GT latent oracle ranking

- Mean rho 0.720, 95% CI [0.672, 0.763], median 0.797.
- Pairwise accuracy 0.859, CI [0.834, 0.883].
- Mean regret 0.0273, CI [0.0205, 0.0346].

## 10. Predicted latent ranking

- Mean rho 0.708, 95% CI [0.654, 0.758], median 0.796.
- Pairwise accuracy 0.861.
- Mean regret 0.0311.
- Top-5 retrieval 0.675 versus 0.650 for GT.

## 11. Oracle gap

- Delta rho (GT-PRED): 0.012, CI [-0.034, 0.058].
- Delta regret (PRED-GT): 0.0038, CI [-0.0059, 0.0150].

## 12. Boundary-near versus boundary-far

Boundary-specific excess pair-accuracy degradation is -0.055, CI [-0.143, 0.032]. Adjacent pairs are used in both groups, so action-grid distance is matched.

## 13. Statistical uncertainty

All confidence intervals are 5000-sample state-level bootstraps. Actions and pairs are aggregated within anchor before inference.

## 14. Representative cases

See `plots/ranking_diagnostic/representative_rankings.png`, including four distinct both-strong, largest-gap, predicted-better counterexample, and GT-failure states selected by deterministic metric rules. No state was manually chosen.

## 15. Failure cases

The fixed official goal-L2 scorer fails even with true futures (mean rho -0.046, mean regret 0.121). Individual learned-scorer failures and counterexamples remain in `results/ranking_diagnostic/test/state_metrics.csv`; none was removed.

## 16. Scientific interpretation

The representation contains recoverable decision information, while predicted latents retain essentially the same ranking utility. The fixed official goal-L2 score is the bottleneck that made the prior planner diagnostic weak; predictor fidelity is not the bottleneck under this local protocol.

## 17. Final decision

**PREDICTOR NOT BOTTLENECK**

GT oracle gate: True. Material predictor gap: False. Boundary-specific gap: False. No Boundary-JEPA intervention was trained.
