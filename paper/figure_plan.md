# Figure Plan

## Figure 1 — Controlled bottleneck attribution

Pipeline: representation/readout → prediction → decision metric → search/proposal → selected action. Annotate each oracle substitution and state that headroom terms are not additive.

Current asset: `plots/bottleneck_decomposition/system_decomposition.png`.

## Figure 2 — Push-T bottleneck profile

Prediction, metric, proposal, and within-search selection headroom with 95% state-bootstrap intervals. Explicitly note selection/metric overlap.

Current asset: `plots/bottleneck_decomposition/pusht_bottleneck_profile.png`.

## Figure 3 — Push-T pre-specified repair

Baseline BF16+L2, targeted BF16+readout, and wrong-layer FP32+L2 under identical CEM budgets. Caption must call FP32 an inference-precision control.

Current asset: `plots/bottleneck_decomposition/targeted_repair.png`.

## Figure 4 — Wall independent replication

Preferred compact layout after completion:

- Left: Wall diagnosis bottleneck profile on diagnosis states.
- Middle: preregistered repair versus baseline/wrong-layer on disjoint repair states.
- Right: one deterministic representative and one failure state only if legible.

Current assets:

- `plots/wall_replication/wall_bottleneck_profile.png`
- `plots/wall_replication/wall_repair_validation.png`
- `plots/wall_replication/wall_representative_state.png`
- `plots/wall_replication/wall_failure_state.png`

Use the profile and repair plot in the main paper. Keep representative/failure states as a compact third panel or supplement depending on page limits.

## Supplementary candidates

- Push-T representative state plot.
- Wall replay/label/cost audit.
- Helped/unchanged/harmed improvement distribution.

Do not add panels merely to increase figure count.
