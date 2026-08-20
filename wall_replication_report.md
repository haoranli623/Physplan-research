# Independent Wall Replication Report

## Question

Can controlled oracle substitutions diagnose the limiting interface of a latent world-model planner on a second task, and can that diagnosis prospectively select an effective intervention on states that were never used for development or diagnosis?

## Baseline and feasibility

The experiment uses the repository's official `DotWall` simulator and official epoch-50 Wall JEPA-WM checkpoint. The checkpoint loaded with all predictor and proprio-encoder keys matched. It uses a frozen DINOv2 ViT-S/14 encoder, six-layer AdaLN predictor, H6, frameskip 5, two-dimensional actions, and BF16 inference.

State cloning is exact: independently constructed environments produced state-identical and byte-identical rendered rollouts from the same fixed layout/state/action sequence. The vectorized simulator path was also state- and image-identical to direct stepping.

The fixed reference set contains 41 two-segment, 30-control action sequences parameterized by waypoint height at the wall plane. Simulator cost is terminal Euclidean goal distance in the native 65-pixel workspace. On 30 feasibility-only anchors the median cost range was `27.156` pixels with a median of 38 distinct costs. A true-latent action-blind readout trained on 20 feasibility anchors achieved held-out mean rho `0.8918` on 10 anchors. All prewritten feasibility criteria passed.

## Three-way split

The independent unit is an anchor state.

| Split | Anchors | Seed range start | Role |
|---|---:|---:|---|
| `D_wall_dev` | 60 | 1,000,000 | readout regularization and repair development |
| `D_wall_diagnosis` | 80 | 1,100,000 | frozen component attribution |
| `D_wall_repair` | 80 | 1,200,000 | unseen intervention validation |

Seed ranges do not overlap. Each split also uses six layout tuples; pairwise layout-tuple intersections are empty. Repair features did not exist when the diagnosis and repair prediction were committed.

## Frozen protocol

Protocol-freeze commit: `71e019f17df524602c9a5db89205da161a4e372a`.

Protocol SHA256: `058852BEC36525F3ACB69F0E21D9F231F60A84427D5792ED3A2BE2A3E7BF0544`.

The small readout is standard scaling plus ridge (`alpha=10`), chosen by five-fold anchor-grouped GT-latent Spearman on `D_wall_dev`. It receives signed, absolute, and squared terminal-to-goal differences of pooled visual and proprio latents and never receives action. It is trained only on true dev latents and is not refit on predicted latents.

Planning uses the same 11 fixed query indices for every method. Primary regret is divided by each state's 41-candidate cost range. All CIs are 5,000-sample state bootstraps.

## Frozen diagnosis

The diagnosis evaluator was committed in `960bcd0` before generation. Results on 80 diagnosis states were:

| Quantity | Mean | 95% CI |
|---|---:|---:|
| GT readout rho | 0.9428 | [0.9149, 0.9622] |
| GT readout normalized regret | 0.0036 | [0.0012, 0.0067] |
| Prediction gap | 0.5636 | [0.4871, 0.6373] |
| Decision-metric gap | -0.0279 | [-0.0905, 0.0345] |
| Search/proposal gap | 0.0333 | [0.0131, 0.0594] |
| Within-search selection gap | 0.4924 | [0.4228, 0.5650] |

The frozen rule classified the bottleneck as **PREDICTION**. Selection loss is reported but not treated as additive; it is downstream of the predictor/metric scores.

## Pre-registered repair prediction

`wall_bottleneck_prediction.json` and diagnosis artifacts were committed in `8559594b94846b33e84292b60dfdb308bb927b18` before repair states were opened.

The targeted repair was fixed in advance: start from the official checkpoint and load the predictor-only checkpoint trained on 48 dev anchors for five epochs with the original visual latent L2 plus 0.1 proprio L2. Epoch count was selected only by latent L2 on 12 held-out dev anchors. Encoder, latent L2 decision metric, candidates, query indices, horizon, and simulator cost remain unchanged.

The wrong-layer control applies the frozen task readout to the official BF16 predicted futures. It does not modify the predictor.

## Independent repair validation

| Method | Mean normalized regret | 95% CI | Mean pixel regret |
|---|---:|---:|---:|
| Official predictor + L2 | 0.5561 | [0.4895, 0.6250] | 15.1163 |
| Predictor repair + L2 | 0.1670 | [0.0992, 0.2402] | 3.9879 |
| Wrong-layer metric readout | 0.5492 | [0.4809, 0.6194] | 15.2391 |

Targeted improvement was `0.3891` `[0.2930, 0.4804]`; wrong-layer improvement was `0.0069` `[-0.0489, 0.0625]`; their paired contrast was `0.3822` `[0.2797, 0.4798]`. All three preregistered success checks passed.

The targeted repair helped 62 states (77.5%), was unchanged on 7 (8.75%), and harmed 11 (13.75%). The wrong-layer control helped 17, was unchanged on 49, and harmed 14.

As a secondary diagnostic, not a success criterion, the mean full-trajectory visual latent MSE fell from `0.8564` to `0.1999`, a reduction of `0.6565` `[0.6050, 0.7075]`.

## Representative and failure states

Anchor 53 is the median helped example: normalized regret changed `0.6508 -> 0.0000008`; the wrong-layer readout selected a candidate with normalized regret `1.0`.

Anchor 31 is the strongest harmed counterexample: regret changed `0.2150 -> 0.9905` even though its average latent MSE fell `0.7860 -> 0.1986`. This is important evidence against the stronger claim that lower average prediction loss guarantees better planning on every state.

## Interpretation

Push-T and Wall do not share the same bottleneck. Push-T had small prediction headroom and large metric headroom; Wall had large prediction headroom and no positive metric headroom. The same diagnosis-first procedure selected metric repair for Push-T and predictor repair for Wall, and the independently selected Wall repair substantially outperformed its wrong-layer control.

The supported contribution is therefore the workflow—controlled attribution, committed repair prediction, and unseen validation—not a universal claim about JEPA predictors or latent L2.

## Limitations

- Wall uses a controlled one-dimensional waypoint family, not the official continuous full-sequence CEM planner.
- Only one predictor-repair training seed was run.
- The readout ceiling is a joint representation/readout test, not pure representation identifiability.
- Headroom substitutions interact and are not additive causal effects.
- Disjoint layouts strengthen separation but introduce mild distribution shift between splits.
- Thirteen-and-three-quarter percent of repair states were harmed.
