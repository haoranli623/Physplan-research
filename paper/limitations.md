# Limitations and Claim Boundaries

## Scope of completed evidence

- One action-conditioned JEPA world-model family is evaluated.
- Two simulated tasks are complete: Push-T and Wall.
- The local Push-T and Wall protocols use controlled low-dimensional candidate families; the official Push-T H6 action-sequence protocol is now complete on 30 states.
- Only Wall has disjoint development, diagnosis, and repair state sets.
- The official Push-T dataset-sourced initial states were unavailable; the H6 validation uses a frozen fresh near-contact generator and canonical goal.

## Attribution limits

- Headroom terms are paired controlled contrasts, not additive causal components.
- A component's apparent headroom can change after another component is replaced.
- GT readout combines representation observability and readout capacity; it is not a pure representation oracle.
- The learned readout uses simulator task costs on development states.
- A near-zero mean prediction gap does not imply that every state is predicted adequately.
- Full-sequence fixed-trace headroom and adaptive-CEM repair measure different interventions because the scorer changes subsequent proposals.

## Intervention limits

- Push-T diagnosis and repair use the same final state set. The intervention was temporally pre-specified, but it is not independently validated on unseen states.
- Push-T's wrong-layer control is increased inference precision (BF16 to FP32), not retraining, new data, or a new predictor architecture.
- Wall's predictor repair uses one training seed.
- Training compute is not matched between the Wall predictor repair and lightweight metric control.
- Eleven of 80 held-out Wall states are harmed by the predictor repair.
- The full-sequence metric repair helps/does not change/harms 16/9/5 states, and its 95% CI crosses zero.
- No trained full-sequence predictor repair has yet tested the frozen `PREDICTION BECOMES IMPORTANT` diagnosis.

## Cross-task limits

- Push-T and Wall use different physical costs, candidate geometries, and normalizations.
- Absolute gap magnitudes should not be compared as a common effect scale.
- Two tasks do not establish universal bottleneck taxonomy or architectural generality.
- The 30-state full-sequence sample yields wide intervals and is not an architecture-level replication.
- Different task diagnoses show that the procedure is not mechanically anti-L2; they do not prove it will identify a unique dominant component everywhere.

## Language that is supported

> Under the frozen Push-T local protocol, decision-metric headroom dominates prediction headroom, and the pre-specified metric repair recovers substantially more regret than the inference-precision control.

> Under the pre-registered Wall protocol, prediction headroom dominates; a pre-specified predictor repair improves regret on unseen states, while the wrong-layer metric repair does not.

> Controlled oracle substitutions can identify intervention-relevant recoverable headroom in these two studies.

> Under official Push-T H6 sequence planning, prediction headroom exceeds metric headroom; the local metric repair has favorable but inconclusive mean benefit because adaptive search coverage changes.

## Language that is not supported

- “JEPA world models are generally metric limited.”
- “Prediction error does not matter for planning.”
- “The decomposition uniquely or causally assigns all error.”
- “L2 is always a poor planning metric.”
- “Lower prediction loss guarantees better control.”
- “The local metric bottleneck generalizes to realistic MPC.”
- “The full-sequence metric repair is validated” when its confidence interval crosses zero.
- “The post-hoc state-wise correlation is causal or deployable.”
- “FP32 tests predictor improvement” without the qualifier “inference precision.”
- Any “first” or state-of-the-art claim.
