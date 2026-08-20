# Boundary-JEPA Secondary Diagnostic

This report is secondary. The locked primary pilot remains **INCONCLUSIVE**.

- Horizon: 2 latent steps / 10 controls.
- Same-anchor extension: False.
- Predicted-latent planner mean regret: 0.1080.
- True-latent planner-floor mean regret: 0.1008.
- Mean predictor incremental task cost: 0.0072.
- Model/true-latent selectors equal simulator oracle on
  1.0% /
  1.0% of states.
- End-to-end boundary error vs regret: Spearman
  0.089, 95% CI
  [-0.107, 0.275].
- Predictor-vs-GT-probe boundary error vs regret: Spearman
  0.142, 95% CI
  [-0.066,
  0.332].
- Clean single-crossing maps in both GT/pred probe: 38/100.

Interpretation: the true-future-latent planner floor is close to predictor-planner
regret, so the locked regret is predominantly limited by goal representation,
objective, and candidate alignment. This does not rescue Q3 and does not authorize
intervention training.
