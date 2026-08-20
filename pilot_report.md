# Boundary-JEPA Push-T Feasibility Pilot

## 1. Research hypothesis

Pointwise latent prediction accuracy can miss decision-critical local counterfactual contact-boundary misalignment.

## 2. Environment and baseline

- Official Meta JEPA-WMs Push-T implementation and official epoch-50 checkpoint.
- Frozen DINOv2 ViT-S/14 target encoder and official six-layer AdaLN action-conditioned predictor.
- Two predicted latent steps; each latent step concatenates five simulator control actions.

## 3. Dataset protocol

- 100 state-disjoint held-out evaluation anchors.
- 41 cloned actions per one-sided angular sweep.
- Anchors were accepted by a predeclared simulator-only rule requiring exactly one contact/no-contact crossing.
- Probe training/calibration anchors are disjoint from evaluation anchors.

## 4. Physical labels

Primary physical label: simulator-native contact/no-contact from collision-point counts. Persistent/lost-contact were recorded separately but were not used as the primary label. Task coverage/cost remained separate.

## 5. Probe design

The primary probe consumes current latent plus two consecutive latent-transition deltas (visual and proprioceptive), never action. It was trained only on ground-truth latents, calibrated on held-out probe anchors, then frozen.

Held-out GT-latent probe balanced accuracy: **0.847**; macro-F1: **0.847**; AUROC: **0.933**. Current-state-only balanced accuracy: **0.500**.

## 6. Boundary metric

Primary metric: absolute threshold-crossing displacement normalized by the angular sweep span. Missing predicted crossings receive the maximum normalized error of 1. Raw grids and probabilities are retained.

## 7. Planning-regret definition

For the same saved candidate set, the model selects the minimum official latent-L2 goal cost and the oracle selects the minimum simulator cost. Regret is their simulator-cost difference.

## 8. Number of anchor states/actions

100 anchors × 41 actions = 4100 held-out counterfactual rollouts.

## 9–12. Primary statistics

- Mean latent MSE: 0.145931 (median 0.141436).
- Mean normalized end-to-end boundary error: 0.243 (median 0.101).
- Mean GT-probe boundary floor: 0.174 (median 0.081).
- Mean local planning regret: 0.1080 (median 0.0560).

## 13. State-level correlations

- Latent error vs regret: Spearman 0.210, 95% bootstrap CI [0.007, 0.390].
- Boundary error vs regret: Spearman 0.089, 95% bootstrap CI [-0.107, 0.280].
- Difference (boundary minus latent): -0.121, CI [-0.376, 0.116].

These are associations across anchor states, not causal claims.

## 14. Representative visualizations

- `plots/pilot/error_vs_regret.png`
- `plots/pilot/representative_boundary_maps.png`
- `plots/pilot/metric_distributions.png`

The four boundary examples are selected deterministically to cover low/high combinations of latent and boundary error, not manually cherry-picked.

## 15. Failure cases

See `results/pilot/state_metrics.csv`, including missing/multiple predicted crossings and all selected/oracle actions. No states were silently removed.

The secondary decomposition (`results/pilot/diagnostic_summary.json`) found:

- The GT-latent probe has exactly one directional crossing on only 42/100 states; the predicted-latent probe has one on 80/100, and both are clean on 38/100. High action-level probe accuracy therefore does not guarantee a monotone boundary curve.
- Mean end-to-end boundary error is 0.243, while the GT-probe floor is already 0.174. The median excess over the probe floor is only 0.014.
- Selecting with **true** future latents still gives mean simulator regret 0.101, versus 0.108 when selecting with predicted latents. Predictor error adds only 0.007 mean task cost under this planner; the dominant limitation is the latent-goal objective/candidate/task alignment.
- Candidate ranking by true-latent goal cost has mean statewise Spearman -0.108 with simulator cost (89 non-constant states). This is a planner/representation floor, not evidence for or against boundary preservation.

### Pre-recorded horizon diagnostic

The official Push-T planning configuration uses six latent steps, while the locked short-contact pilot used two. To test whether the planner floor was merely a short-horizon artifact, the exact same 100 anchors and 41 actions were extended from 10 to 30 simulator controls. All 100 sweeps retained exactly one simulator contact crossing and 12 replay checks had zero failures.

At six latent steps, predicted-latent planner regret is 0.108 and true-latent planner-floor regret is 0.096; predictor incremental task cost is 0.013. Boundary error versus the longer-horizon regret remains weak (Spearman 0.084, 95% bootstrap CI [-0.124, 0.266]). Thus the horizon diagnostic does not rescue Q3. It is secondary and does not replace the primary H2 result or probe.

## 16. Decision

**INCONCLUSIVE**

Q1 probe observability: True. Q2 low-error boundary misalignment: False. Q3 stronger boundary/regret association (point/strong): False/False.

Operationally, this is a **NO-GO for A/B/C/D training under the current action slice and planning diagnostic**. That statement does not relabel the predeclared pilot decision; it follows the protocol instruction not to burn compute on interventions when the required phenomenon has not been established.

## 17. Exact recommended next step

Do not train Boundary-JEPA yet. Prospectively define and validate a new boundary-sensitive local planning setup on fresh development states, with this gate: **the true-future-latent selector must first achieve materially lower regret than the predicted-latent selector and rank simulator outcomes sensibly on the same candidates**. Only then freeze that planner/candidate protocol and rerun Q2/Q3 on new held-out states. Do not tune the existing 100 evaluation states or retroactively replace their primary result.
