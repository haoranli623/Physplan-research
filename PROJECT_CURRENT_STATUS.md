# CENTRAL CLAIM

Latent world-model planning failures should be diagnosed at the prediction, decision-metric, and search interfaces before retraining or rescoring. The dominant recoverable bottleneck changes with task and planner regime; oracle headroom terms are controlled contrasts, not additive causal components.

# PUSH-T RESULT

In the controlled local H6 angular family, prediction gap is 0.0071 [-0.0051, 0.0198] and metric gap is 0.0851 [0.0608, 0.1109]. The precommitted metric repair improves regret by 0.0815 [0.0563, 0.1079], while the FP32 inference-precision control improves by 0.0017 with CI spanning zero.

# WALL RESULT

In the pre-registered Wall diagnosis, prediction gap is 0.5636 [0.4871, 0.6373] and metric gap is -0.0279 with CI spanning zero. On disjoint repair states, the predictor repair improves regret by 0.3891 [0.2930, 0.4804], versus 0.0069 [-0.0489, 0.0625] for the wrong-layer metric repair.

# FULL-SEQUENCE RESULT

Under official Push-T H6 sequence CEM (30 final episodes; 9000 queries per planner), prediction gap is 0.1608 [0.0565, 0.2725] and metric gap is 0.0950 [0.0026, 0.1978]. The frozen decision is `PREDICTION BECOMES IMPORTANT`. Metric repair changes regret from 0.4974 to 0.4131, an improvement of 0.0843 [-0.00008, 0.1684], with 16/9/5 helped/unchanged/harmed states.

# WHAT REPLICATED

The attribution protocol continued to produce an auditable, nontrivial diagnosis on identical retained full-sequence candidates. Residual metric headroom and a favorable mean repair direction replicated qualitatively. Exact CEM behavior, state-level bootstrap, no filtering, and matched query budgets all held.

# WHAT CHANGED UNDER REALISTIC PLANNING

The local metric bottleneck did not remain dominant. Prediction error grew materially across long 60-D action sequences, and changing the scorer also changed CEM proposal coverage. Candidate-matched headroom did not translate cleanly into adaptive-search improvement.

# STRONGEST EVIDENCE

Across three frozen regimes, the same diagnostic story does not mechanically blame one layer: local Push-T is metric dominant, Wall is prediction dominant, and official Push-T H6 becomes prediction important while retaining weaker metric headroom. Both earlier pre-specified repairs validate their diagnoses; the full-sequence metric repair correctly fails the stronger criterion when its CI crosses zero.

# IMPORTANT COUNTEREXAMPLES

Full-sequence episode 20 worsens from 0.4044 to 1.0000 regret even though the targeted scorer selects the simulator-best action within its own trace; the trace itself has coverage gap 0.5811. Wall anchor 31 also worsens despite lower prediction MSE. These cases rule out both “better scoring always improves planning” and “lower prediction loss guarantees better decisions.”

# CURRENT LIMITATIONS

One JEPA-WM family and simulated tasks; local candidate geometries for the strongest prospective repairs; only 30 official full-sequence states; no independent global oracle for adaptive CEM; task readouts require simulator-cost supervision; one Wall repair training seed; and no trained full-sequence predictor repair yet.

# WORKSHOP READINESS

The project is workshop-ready as a controlled bottleneck-attribution paper with a valuable negative transfer result. The strongest narrative is not “L2 is bad,” but “the limiting interface changes with planner regime, and diagnosis prevents overgeneralizing a successful local repair.” Full-paper claims must preserve the frozen intervals and the inconclusive full-sequence repair.

# NEXT REQUIRED EXPERIMENT

Pre-register and run one full-sequence predictor repair under the identical H6 protocol, retaining the metric repair as a wrong-layer control. Prefer an independent reference proposal set if compute permits. This is a post-workshop strengthening experiment, not required to submit the current honest result.

# EXPERIMENTS WE SHOULD NOT DO

Do not add a third task, another scorer, a new JEPA loss, Boundary-JEPA training, broad architecture sweeps, or post-hoc threshold changes before submission. Do not rerun the final 30 states to seek a positive metric-repair interval.
