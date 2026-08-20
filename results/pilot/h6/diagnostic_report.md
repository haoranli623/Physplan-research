# Boundary-JEPA Secondary Diagnostic

This report is secondary. The locked primary pilot remains **INCONCLUSIVE**.

- Horizon: 6 latent steps / 30 controls.
- Same-anchor extension: True.
- Predicted-latent planner mean regret: 0.1082.
- True-latent planner-floor mean regret: 0.0956.
- Mean predictor incremental task cost: 0.0126.
- Model/true-latent selectors equal simulator oracle on
  4.0% /
  4.0% of states.
- End-to-end boundary error vs regret: Spearman
  0.084, 95% CI
  [-0.124, 0.266].
- Predictor-vs-GT-probe boundary error vs regret: Spearman
  0.039, 95% CI
  [-0.179,
  0.248].
- Clean single-crossing maps in both GT/pred probe: 38/100.

Interpretation: the true-future-latent planner floor is close to predictor-planner
regret, so the locked regret is predominantly limited by goal representation,
objective, and candidate alignment. This does not rescue Q3 and does not authorize
intervention training.
