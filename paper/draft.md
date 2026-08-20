# Where Does Latent Planning Fail?

## Controlled Bottleneck Attribution for World-Model Planning

### Abstract

When a latent world-model planner selects a poor action, the failure is commonly attributed to inaccurate prediction. That diagnosis is incomplete: decisions also depend on the latent representation, the cost used to rank predicted futures, and the candidates exposed by search. We introduce a controlled attribution protocol that changes one interface at a time while preserving the candidate set. Simulator-evaluated oracle substitutions estimate recoverable prediction, decision-metric, proposal, and within-search headroom at the level of independent initial states. The protocol is designed to predict the component worth repairing before the intervention is run. On Push-T, predicted and ground-truth latents perform similarly under the same learned task readout (normalized-regret gap 0.0071, 95% bootstrap CI [-0.0051, 0.0198]), while replacing latent L2 with that readout exposes substantially larger metric headroom (0.0851 [0.0608, 0.1109]). A pre-specified metric repair then improves regret by 0.0815 [0.0563, 0.1079], whereas increasing predictor inference precision improves it by only 0.0017 with an interval spanning zero. On a disjoint Wall replication, the same protocol instead diagnoses prediction as limiting (gap 0.5636 [0.4871, 0.6373]); the subsequently evaluated predictor repair improves regret by 0.3891 [0.2930, 0.4804] on unseen states, while the wrong-layer metric repair is near zero. These controlled studies show that different planning problems can occupy different failure regimes and support a practical principle: diagnose the limiting interface before retraining the world model. An official Push-T full action-sequence validation is in progress and is not included in the present claims.

## 1. Introduction

Latent world models make planning computationally attractive by predicting future representations rather than reconstructing pixels. A planner nevertheless contains more than a predictor. The observation must be represented, candidate actions must be proposed, predicted futures must be assigned decision scores, and a candidate must be selected. A low-quality action can therefore arise even when the predicted latent is close to its target under the training loss; conversely, a visibly imperfect prediction can remain sufficient for choosing the best available action.

This distinction matters operationally. Improving a predictor can require new data, substantial accelerator time, and delicate optimization. If the dominant recoverable error lies in the decision metric or search distribution, that investment targets the wrong component. The reverse error is equally possible: learning a more elaborate scorer cannot recover information the predictor has lost.

We study whether controlled oracle substitutions can identify the dominant recoverable bottleneck before a repair is attempted. For a fixed initial state and fixed candidate actions, we compare selections made using ground-truth or predicted latent futures and task-aligned or default scores. Every selected action is then executed in the simulator. The resulting gaps answer intervention-oriented questions: how much could be recovered by fixing prediction while retaining the same scoring interface, or by fixing the scoring interface while retaining the same predictions? Separate simulator oracles over queried and reference candidates expose search coverage and within-search selection.

Our experiments deliberately emphasize controlled evidence over breadth. Push-T supplies a diagnosis-to-intervention study in a local H6 action family. Wall supplies a pre-registered replication with disjoint development, diagnosis, and repair states. The two tasks yield different diagnoses. Push-T is decision-metric limited under the frozen protocol; Wall is prediction limited. In both cases, the repair selected by the diagnosis substantially outperforms a task-appropriate wrong-layer control.

The contributions are:

1. A candidate-matched, simulator-grounded protocol for estimating recoverable headroom at prediction, decision-metric, proposal, and selection interfaces of a latent planner.
2. Prospective evidence on Push-T that the diagnosed metric repair recovers the predicted headroom while a predictor-precision control does not.
3. An independent Wall replication in which the same rule selects a different bottleneck and the corresponding predictor repair succeeds on unseen states.
4. A claim ledger and artifact manifests that explicitly delimit what these two controlled studies do and do not establish.

We do not propose a new JEPA architecture, latent loss, or optimizer. The contribution is diagnostic: a planning failure should not automatically be labeled a prediction failure.

## 2. Controlled Bottleneck Attribution

### 2.1 Planning pipeline

Let an encoder map the current observation and a goal observation to latent representations. Given state representation \(z_s\) and candidate action sequence \(a\), a predictor produces \(\hat z(a)\). A decision metric \(d(\hat z(a),z_g)\) ranks candidates generated by a proposal or search distribution. The simulator cost \(J_s(a)\), which is unavailable to the deployed planner, is used only for evaluation. Lower cost is better.

For each initial state \(s\), let \(A_s\) be a fixed reference candidate set. A method selects

\[
  a_m(s)=\arg\min_{a\in A_s} q_m(s,a),
\]

and its finite-set normalized regret is

\[
  R_m(s)=\frac{J_s(a_m)-\min_{a\in A_s}J_s(a)}
  {\max_{a\in A_s}J_s(a)-\min_{a\in A_s}J_s(a)+\epsilon}.
\]

All headline estimates use the initial state as the independent unit. Nearby actions from the same state are never counted as independent trials.

### 2.2 Controlled substitutions

The main comparisons preserve the state, candidate actions, task readout, and simulator evaluation while substituting one latent source or score:

- **GT readout:** a frozen task readout scores encoded simulator futures.
- **Predicted readout:** the same readout scores predicted futures.
- **Predicted L2:** the baseline latent distance scores the same predicted futures.
- **Simulator oracle:** the simulator chooses the best candidate, either among all reference actions or among actions actually queried by search.

The prediction and metric headroom estimates are

\[
  G_{\mathrm{pred}} = \mathbb{E}_s[R_{\mathrm{pred\text{-}readout}}(s)-R_{\mathrm{GT\text{-}readout}}(s)],
\]

\[
  G_{\mathrm{metric}} = \mathbb{E}_s[R_{\mathrm{pred\text{-}L2}}(s)-R_{\mathrm{pred\text{-}readout}}(s)].
\]

Proposal headroom compares the simulator-best queried candidate with the best reference candidate. Within-search selection headroom compares the planner-selected queried candidate with the simulator-best queried candidate. The terms are controlled contrasts, not an additive causal decomposition: interfaces interact, and the learned readout itself has finite capacity.

The task readout never sees the action. It maps latent transition information to simulator task cost and is fit only on development states. This prevents a direct action-to-outcome shortcut. Simulator labels and costs are unavailable to the world-model predictor.

### 2.3 Decision rule and validation logic

Before each repair evaluation, a protocol fixes the substitutions, dominance margin, bootstrap procedure, intervention choice, wrong-layer control, and success criterion. The diagnosed component is the eligible gap whose lower confidence bound clears the frozen margin and dominates other eligible gaps. If no component satisfies the rule, the correct output is inconclusive.

This logic makes a stronger test than explaining an intervention after observing it. The prediction is directional: a targeted repair should outperform both the unchanged baseline and a repair at a non-dominant layer. The repair need not help every state, and all harmed states remain in the analysis.

## 3. Push-T: Decision-Metric Bottleneck

### 3.1 Frozen local protocol

The Push-T study uses the official action-conditioned JEPA world model and simulator interface. Evaluation contains 100 fresh cloned anchor states. Each state has the same 81-action local angular reference family at horizon H6. A finite-grid CEM-like planner receives 65 queries per state. The frozen task readout is an action-blind ridge model trained on development states using ground-truth latent transitions and simulator costs. Exact cloned replay and action/temporal alignment were validated before attribution.

This local family is intentionally controlled: it exposes many nearby alternatives from exactly the same physical state and makes candidate matching auditable. It is not a claim about arbitrary high-dimensional model-predictive control.

### 3.2 Diagnosis

The GT and predicted task readouts achieve state-level Spearman correlations of 0.696 and 0.689 with simulator cost, with normalized regrets of 0.0289 and 0.0360. Their difference yields \(G_{\mathrm{pred}}=0.0071\) with 95% state-bootstrap CI [-0.0051, 0.0198]. Thus substituting ground-truth for predicted latent futures under the same readout offers little recoverable headroom under this protocol.

By contrast, the default latent-L2 decision score produces \(G_{\mathrm{metric}}=0.0851\) [0.0608, 0.1109]. Proposal headroom is 0.0074. Under the frozen dominance rule, the decision metric is the bottleneck. Importantly, this is not the claim that predicted latents are perfect: individual prediction failures remain, but correcting them is not the largest available average intervention in this candidate family.

### 3.3 Prospective intervention and control

The diagnosis and proposed repair were committed before repair computation. With the same 65-query budget, replacing latent L2 with the frozen task-aligned readout reduces normalized regret from 0.1237 to 0.0421, an improvement of 0.0815 [0.0563, 0.1079]. As a wrong-layer control, predictor inference was changed from BF16 to FP32 while retaining latent L2. This changes regret by only 0.0017 and its confidence interval spans zero.

The narrow interpretation is important. FP32 is an inference-precision control, not a generally stronger predictor trained with additional data. The result shows that increasing numerical precision did not recover the metric headroom predicted by the decomposition. It does not prove that no possible predictor improvement could help.

The intervention follows the diagnosed effect size closely: the measured repair gain of 0.0815 is comparable to the prior metric headroom of 0.0851, whereas the control is comparable to the near-zero prediction headroom. This agreement is the principal Push-T evidence that the substitutions are intervention-relevant rather than merely descriptive.

## 4. Wall: Independent Prediction-Bottleneck Replication

### 4.1 Pre-registered split and protocol

Wall tests whether the framework is hard-coded to prefer a learned metric. It uses the official environment/model interface and a controlled one-dimensional waypoint family. Exact clone replay, physical labels, cost monotonicity, and model rollout were audited before scaling. The data are divided into 60 development, 80 diagnosis, and 80 repair states with disjoint seeds, cloned states, and layout tuples. The protocol was committed before diagnosis, and the repair prediction was committed while the repair cache was absent.

### 4.2 A different failure regime

On the diagnosis states, prediction headroom is \(G_{\mathrm{pred}}=0.5636\) [0.4871, 0.6373]. Metric headroom is -0.0279 with a confidence interval spanning zero, and proposal headroom is 0.0333. The same frozen rule that labels Push-T as decision-metric limited therefore labels Wall as prediction limited.

This contrast is scientifically useful. If the protocol merely encoded an objection to latent L2, it would have produced the same diagnosis on both tasks. Instead, it distinguishes a regime in which downstream rescoring cannot restore information lost by the predictor.

### 4.3 Repair on unseen states

The pre-specified predictor repair is trained only from development data and evaluated on the 80 unseen repair states. It reduces normalized regret from 0.5561 to 0.1670, an improvement of 0.3891 [0.2930, 0.4804]. The wrong-layer metric repair improves regret by 0.0069 [-0.0489, 0.0625]. The targeted-minus-control contrast is 0.3822 [0.2797, 0.4798]. At the state level, the targeted repair helps 62 states, is unchanged on 7, and harms 11; those harmed states are retained.

As a mechanism check, latent prediction MSE decreases from 0.8564 to 0.1999. Lower MSE is not sufficient for a better decision on every state: on one retained failure case (anchor 31), planning regret increases from 0.215 to 0.991 despite the MSE improvement. The aggregate repair result therefore supports the diagnosis, while the failure case prevents the stronger—and unsupported—claim that improving a predictive loss guarantees better planning.

## 5. Official Full Action-Sequence Planning

The preceding experiments use controlled local candidate families. A focused validation with the official Push-T H6 action-sequence CEM planner is being developed under a separately frozen protocol. It will retain every candidate sequence, compare GT-readout, predicted-readout, and predicted-L2 rankings on identical fixed traces, and evaluate a matched-query adaptive CEM repair that changes only the scorer. Development episodes are separated from the final episode set, and the episode remains the bootstrap unit.

No full-sequence result is reported here yet. In particular, the local Push-T result is not presented as evidence that the official iterative planner is metric limited. This section will be updated only after the final protocol is committed and the new episodes are evaluated.

## 6. Related Work

Action-conditioned latent world models plan without pixel reconstruction by predicting features from pretrained or self-supervised encoders. DINO-WM demonstrated zero-shot planning over frozen DINOv2 features, while V-JEPA 2 and JEPA-WMs study action-conditioned joint-embedding prediction and physical planning at larger scale. These works analyze representation, prediction, objective, or planner choices; our focus is a state-level controlled substitution protocol that prospectively selects which interface to repair.

Decision-aware model learning has long challenged the idea that generic predictive accuracy is the sole objective. Value-aware loss functions and value-targeted regression optimize models according to downstream value structure, and temporal-distance representations learn geometry suited to goal reaching. We do not claim that task-aligned scores are new. Their role here is as an oracle-like diagnostic and a deliberately simple repair whose choice is fixed before evaluation.

CEM is a standard sampling optimizer for continuous control. Search can fail because its proposal distribution never covers good actions or because the planner misranks actions it did query. Retaining all queried candidates and evaluating them with the simulator lets us separate these cases without introducing a new optimizer.

The closest methodological connection is to component-wise oracle substitution and upper-bound analysis. Our emphasis is on paired candidate sets, state-level inference, explicit non-additivity, and the diagnosis-to-committed-intervention sequence. We avoid priority claims because related diagnostic protocols appear across model-based control and systems evaluation.

## 7. Limitations

The evidence covers one latent world-model family and two controlled simulated tasks. The Push-T diagnosis and repair share the same final states, although the repair was committed before its computation; only Wall provides a disjoint repair-validation set. Both completed studies use controlled low-dimensional candidate geometries, so neither establishes the dominant bottleneck under the official high-dimensional iterative planner.

The GT-readout condition combines representation observability with the finite capacity and development distribution of the ridge readout. It is therefore a practical ceiling, not a pure representation oracle. The readout uses simulator costs during offline development and is not an unsupervised replacement objective. Headroom terms overlap and can change when another interface changes; they must not be summed or treated as a unique causal partition.

The Push-T wrong-layer control changes inference precision, not predictor training, architecture, or data. The Wall predictor repair uses one training seed, and its training cost is not matched to fitting the lightweight metric readout. Candidate families and cost normalizations differ across tasks, so cross-task gap magnitudes should not be compared as if measured on a common physical scale.

Finally, average improvements conceal heterogeneous states. The Wall repair harms 11 of 80 held-out states, and lower prediction MSE does not guarantee lower regret. Deployment would require uncertainty estimates or a fallback rule, neither of which is studied here.

## 8. Conclusion

Planning through a latent world model is a pipeline, not a single prediction loss. Controlled candidate-matched substitutions identify little predictor headroom but substantial decision-metric headroom on Push-T, and the prospectively selected metric repair recovers that gap. The same protocol identifies the opposite regime on Wall; a predictor repair then succeeds on disjoint states while the metric control does not. The supported conclusion is deliberately bounded: under these frozen protocols, attribution predicted which component offered recoverable decision quality. The practical lesson is broader but still a hypothesis for further testing—diagnose before retraining the world model.

## References (working links)

- Terver et al., *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?* [arXiv:2512.24497](https://arxiv.org/abs/2512.24497).
- Zhou et al., *DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning.* [arXiv:2411.04983](https://arxiv.org/abs/2411.04983).
- Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning.* [Meta AI](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/).
- Farahmand et al., *Value-Aware Loss Function for Model-based Reinforcement Learning.* [AISTATS 2017](https://proceedings.mlr.press/v54/farahmand17a.html).
- Ayoub et al., *Model-Based Reinforcement Learning with Value-Targeted Regression.* [ICML 2020](https://proceedings.mlr.press/v119/ayoub20a.html).
- Park et al., *TLDR: Unsupervised Goal-Conditioned RL via Temporal Distance-Aware Representations.* [arXiv:2407.08464](https://arxiv.org/abs/2407.08464).
- Rubinstein, *The Cross-Entropy Method for Combinatorial and Continuous Optimization.* [DOI](https://doi.org/10.1023/A:1010091220143).
