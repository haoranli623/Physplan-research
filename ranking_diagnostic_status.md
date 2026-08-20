# Local Counterfactual Ranking Diagnostic Status

Last updated: 2026-08-20 11:50 EDT

## Current stage

Complete. Fresh held-out evaluation, audits, figures, reports, reproducibility checks, and compute diagnostics are finished. Final frozen decision: **PREDICTOR NOT BOTTLENECK**.

## Starting repository state

- Branch: `boundary-jepa-overnight`
- Starting HEAD: `aa77227`
- Working tree was clean.
- The previous 100-anchor H6 cache has influenced prior reasoning and is therefore development-only.
- Official epoch-50 Push-T JEPA-WM, exact cloned replay, H6 unroll, float64 action/state caches, simulator labels, and resumable HDF5 feature infrastructure are already validated and will be reused.

## Exact existing data flow

1. A seven-dimensional float64 Push-T state is installed through `reset_to_state` in a fresh seeded simulator.
2. Each local candidate is a two-dimensional relative action, held constant for 30 controls (six latent steps × five controls).
3. Simulator endpoints are rendered at latent-step boundaries; `J_sim = 1 - final_coverage`. Contact labels and task outcomes remain separate.
4. The frozen official DINOv2 ViT-S/14 target encoder maps actual future observations to `z_GT`.
5. The official AdaLN action-conditioned predictor receives the same initial latent and normalized five-control action chunks to produce `z_pred`.
6. The previous planner used fixed terminal latent L2 to a canonical goal plus 0.1 proprio L2. Applying this score to true latents was weak, motivating a small development-only readout.

## Concise execution plan

1. Quantify within-state H6 simulator-cost resolution on the old 100 development anchors and retain the fixed official goal scorer as the no-training baseline.
2. Cache a spatially preserving, compact latent descriptor if global mean pooling is insufficient. Do not expose action to the scorer.
3. On development anchors only, compare at most two scientifically motivated simple readouts: endpoint latent first, short trajectory only if endpoint fails the GT-oracle gate. Select regularization with anchor-grouped validation.
4. Predeclare the informative-anchor rule, pair tie tolerance, top-k, boundary-near definition, scorer, horizon, candidate geometry, and state-level metrics.
5. Commit the frozen protocol before generating a fresh seed-disjoint test cache.
6. Generate and audit fresh test anchors using simulator-only acceptance; encode GT/pred latents once; apply the identical frozen scorer to both.
7. Report per-state Spearman, pairwise accuracy, top-k retrieval, selection regret, oracle gaps, state bootstrap confidence intervals, and boundary-near/far degradation.
8. Produce raw CSV/JSON, deterministic success/failure/counterexample plots, the requested report/summary/docs, tests, and final commits.

## Integrity constraints

- Development anchors: existing H6 pilot cache only (plus simulator-only development extensions if explicitly logged).
- Fresh test anchors: unavailable until the freeze commit; never used for scorer/horizon/action-slice/threshold selection.
- Anchor state is the independent unit. Action pairs are aggregated within anchor before confidence intervals.
- The same fitted scorer is applied unchanged to GT and predicted latents.
- No Boundary-JEPA training, A/B/C/D, L_rel tuning, action input, large model, or post-test protocol revision.

## Provisional choices to validate on development data

- Horizon: official H6 / 30 controls.
- Candidate geometry: existing interpretable 41-action one-sided angular sweep with one simulator contact boundary.
- Primary learned scorer family: linear ridge regression on frozen latent representation, fit only to GT latent / simulator cost on development anchors.
- Primary top-k: 5 of 41 candidates.
- Test target: approximately 120 informative, fresh anchors if development throughput and acceptance rate support it.

These remain provisional until the explicit protocol-freeze commit.

## Development iteration 1 — observed before freeze

- Data: the prior 100-anchor H6 cache only; deterministic 70/30 anchor split for the initial sanity check.
- Simulator-cost range quantiles: min 0, Q25 0.056, median 0.208, Q75 0.342, max 0.749. Seventy-six states have range ≥0.05; eleven are constant.
- Adjacent nonzero absolute cost differences have median 0.0133. A numerical tie tolerance of `1e-4` retains 76/100 true boundary-crossing adjacent pairs.
- Existing fixed goal-L2 scorer remains weak on the 30 held-out development states: GT mean rho -0.039 and mean regret 0.069.
- Endpoint global-mean-pooled visual latent plus pooled proprio, StandardScaler + linear Ridge, is sufficient: across the small alpha sweep, held-out GT mean rho is approximately 0.70–0.72 and mean regret 0.014–0.020. Applying the same scorer to predicted latents gives mean rho approximately 0.60–0.68.
- A flattened six-step trajectory readout did not materially improve the GT oracle over the simpler endpoint readout and adds temporal dimension/regularization ambiguity. It is rejected before test generation.

## Decisions logged after development iteration 1

- Freeze candidate horizon at official H6 and candidate geometry at the existing 41-action one-sided sweep.
- Define an informative anchor as exactly one simulator contact crossing and `max(J_sim)-min(J_sim) >= 0.05`. This rule uses simulator outcomes only and will be applied uniformly during test generation.
- Use endpoint pooled latent + endpoint pooled proprio only; scorer never reads action or simulator state.
- Choose Ridge alpha by five-fold anchor-grouped development CV maximizing mean per-state GT Spearman. Fit the final scorer on all development anchors after selection.
- Pairwise tie tolerance is `1e-4`; top-k is 5.
- Boundary-specific comparison uses equal-action-distance adjacent pairs: the single true contact-crossing pair versus same-mode adjacent pairs at least five grid cells from the boundary. Pair correctness is aggregated within anchor before bootstrap.
- No additional readout family or horizon iteration is authorized unless the formal grouped-CV implementation contradicts the sanity check.

## Formal development result and freeze declaration

- Five-fold anchor-grouped out-of-fold selection chose Ridge `alpha=1000` by the predeclared maximum mean GT-rho rule.
- On all 76 informative development states, out-of-fold GT/PRED mean rho is 0.731/0.720, pairwise accuracy 0.866/0.867, top-5 retrieval 0.697/0.697, and mean regret 0.0231/0.0367.
- Development delta rho is 0.011 and delta regret is 0.0136. Near-boundary degradation is not larger than far degradation on development data.
- Frozen scorer SHA256: `2E611EB94852AA588F6C38A2FED9F7472AD6953B0C35817241942A0F1ADC004C`.
- Frozen test seed offset: 4,000,000; target: 120 accepted states. Acceptance is simulator-only: one contact boundary and task-cost range ≥0.05.
- Frozen practical gates are committed in `configs/ranking_diagnostic/protocol.yaml`: GT lower-CI rho >0.5, GT lower-CI pairwise accuracy >0.7, GT upper-CI regret <0.05; material predictor gaps are delta-rho ≥0.10 or delta-regret ≥0.02 with lower CI >0; boundary specificity requires near-minus-far degradation ≥0.05 with lower CI >0.
- Test evaluation is one shot. No threshold, feature, scorer, horizon, candidate rule, or metric may change after the freeze commit.

## Freeze commit

- Commit: `428c2db` (`boundary-jepa: freeze local ranking diagnostic protocol`).
- No fresh `ranking_test` state existed before this commit.

## Fresh held-out test execution

- Generation seed: 24,260,820 (base + frozen 4,000,000 offset).
- 120 accepted anchors from 241 attempts; 69 no-boundary rejections, 52 cost-range rejections, zero multi-boundary rejections.
- Every accepted anchor satisfies the frozen single-boundary and cost-range ≥0.05 rules.
- Twelve exact cloned replays: zero contact/state/coverage failures.
- Feature cache: 120×41 H6 candidates in 182.39 s, 26.98 candidate rollouts/s; peak CUDA allocated/reserved 2.60/2.93 GB.
- The frozen evaluator was invoked once after cache completion. No scorer refit or test-dependent threshold change occurred.

## Frozen test result

- GT latent: mean rho 0.720, 95% state-bootstrap CI [0.672, 0.763]; pairwise accuracy 0.859 [0.834, 0.883]; mean regret 0.0273 [0.0205, 0.0346]. The GT oracle gate passes.
- Predicted latent: mean rho 0.708 [0.654, 0.758]; pairwise accuracy 0.861 [0.833, 0.887]; mean regret 0.0311 [0.0216, 0.0416].
- Oracle gap: delta rho (GT-PRED) 0.012 [-0.034, 0.058]; delta regret (PRED-GT) 0.0038 [-0.0059, 0.0150]. Both upper bounds remain below the frozen material-gap thresholds.
- Boundary-specific near-minus-far pair degradation: -0.055 [-0.143, 0.032]. There is no evidence that prediction selectively loses ranking fidelity near contact boundaries.
- Fixed official goal-L2 scorer remains weak even on GT futures (mean rho -0.046, mean regret 0.121), confirming that the earlier failure was a scoring-interface floor rather than missing decision information in the representation.
- Final decision: **PREDICTOR NOT BOTTLENECK**. This rules out Boundary-JEPA intervention training under the frozen local ranking protocol.

## Post-result engineering robustness (does not replace canonical test)

- The canonical batch=1 feature cache used only 2.60/2.93 GB peak allocated/reserved VRAM, so a resumable multi-anchor path with automatic OOM backoff was added after the scientific result was fixed.
- Batch=5 completed without OOM at 12.40/13.78 decimal GB allocated/reserved (about 80% of 16 GiB), but throughput improved only from 26.98 to 27.29 candidates/s because preprocessing/transfer dominates.
- A batch=6 attempt reached approximately 15,900 MiB total GPU use (about 90% after subtracting baseline desktop use) but completed only 60/120 anchors after more than nine minutes. It was explicitly terminated, renamed `test_features_batch6_aborted_at_60.h5`, and never used scientifically. This confirms that the requested occupancy range is counterproductive on this Windows/BF16 path.
- GT encodings were exact across batch sizes; BF16 predicted features changed slightly (mean absolute visual difference 0.00348). The secondary batch=5 result remained GT/PRED rho 0.720/0.708 and regret gap 0.0035.
- Canonical batch=1 artifacts and their original test summary are retained. The presentation-only representative plot rerender left the scientific summary SHA256 unchanged (`76C149...B7B7BB`).

## Next automatic step

Handoff. No further scientific experiment or Boundary-JEPA training is justified under this frozen protocol.
