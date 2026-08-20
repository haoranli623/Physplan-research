# Bottleneck Decomposition Status

Last updated: 2026-08-20 14:00 EDT

## Current stage

P0-P3 are complete. The implementation and fresh development experiment are complete; the final-test protocol is frozen in the working tree and awaits its required pre-test commit. No final-test state has been generated.

## Starting state

- Branch: `boundary-jepa-overnight`
- Starting HEAD: `ce89c51` (`docs: record ranking compute utilization audit`)
- Working tree was clean.
- Official public Push-T checkpoint: `artifacts/checkpoints/jepa_wm_pusht.pth.tar`.
- Machine: RTX 3080 Laptop GPU, 16 GiB VRAM; 32 GiB RAM; 16 logical CPUs.
- Boundary-JEPA is retired for this Push-T protocol. No Boundary-JEPA, `L_rel`, ensemble, large policy, or world-model training is authorized.

## Verified prior evidence

- The old contact-transition probe is observable from true latent transitions (balanced accuracy/macro-F1 0.847; AUROC 0.933), but boundary-specific predictor degradation was not supported.
- On 120 prospectively frozen held-out local H6 states, the same action-blind endpoint ridge achieved GT/PRED mean rho 0.720/0.708 and regret 0.0273/0.0311.
- The predictor gap was small: delta rho 0.012, CI [-0.034, 0.058]; delta regret 0.0038, CI [-0.0059, 0.0150].
- The official fixed latent-goal L2 score was poor even on true futures (mean rho -0.046; regret 0.121). This motivates a unified downstream decomposition, not another predictor loss.
- Canonical GPU feature caching remains batch 1. Batch 5 gave only ~1.2% throughput gain and batch 6 caused severe slowdown.

## Audited existing data flow

1. A float64 seven-dimensional Push-T state is restored exactly with `reset_to_state`.
2. A two-dimensional relative action is repeated for 30 controls (official H6, five controls per latent step).
3. The simulator saves H6 observations/states, contact labels, final coverage, and `J_sim = 1 - final_coverage` separately.
4. The frozen official DINOv2 target encoder produces true future visual/proprio latents.
5. The official frozen AdaLN predictor consumes the same initial latent and normalized action chunks to produce predicted futures.
6. The prior action-blind `StandardScaler + Ridge(alpha=1000)` consumes terminal pooled visual/proprio latent only and is applied identically to GT and predicted latents.
7. The official baseline planner is CEM. Its published Push-T configuration uses 30 iterations, 300 samples, 10 elites, H6, and latent goal L2. That full 12-D sequence search is not directly commensurate with the existing one-action counterfactual slice.

## Concise execution plan

1. Generate completely new development anchors using a new seed range and a denser fixed one-dimensional angular reference set. Preserve all actions and simulator outcomes.
2. Fit/select only the existing simple endpoint ridge family on development GT latents. Use anchor-grouped CV and do not expose action.
3. Implement a local CEM over the same scalar angular coordinate. It repeats each sampled two-dimensional action for H6, records every queried action, and uses the official frozen predictor. This cleanly tests proposal coverage without introducing a new planner family.
4. On development data, choose one query budget and freeze the reference set, readout, cost definition, bootstrap, dominance rule, and search settings.
5. Commit the protocol before generating fresh final-test anchors.
6. Run the frozen reference-set decomposition and the frozen baseline L2-CEM search on final test states.
7. Save and commit `bottleneck_predictions.json` before any repaired search run.
8. If the diagnosis is decision-metric limited, run matched-budget CEM with the frozen task readout. Use full-precision predictor inference as the cheap wrong-layer control only if a smoke test confirms the same pipeline is stable; otherwise record that no clean control exists.
9. Report controlled headroom terms as non-additive because prediction, metric, and proposal interact.

## Locked assumptions before development

- Push-T remains the only task until its decomposition is technically clean and attributes a meaningful bottleneck.
- Anchor state is the independent unit. Candidate actions are never treated as independent samples for confidence intervals.
- `J_sim = 1 - final_coverage` remains the simulator cost; lower is better.
- The reference and search parameterization is deliberately local: one scalar angle mapped to a constant-magnitude 2-D action repeated for H6. It is a controlled local planning diagnostic, not a reproduction of the official full-sequence benchmark.
- Every search query, its score, action, iteration, simulator outcome, and final selected action will be saved.
- Search and reference candidate sets may differ, so `G_search` is a finite-set coverage estimate and can be slightly negative if CEM finds an action between reference grid points. It will not be clipped or presented as an additive causal term.
- The readout is trained only on new development GT latents. Final test states will not affect scorer, normalization, cost, candidate geometry, dominance thresholds, or search budget.

## Provisional development choices

- Development target: 80 accepted anchors; reference set: 81 angles across the existing one-sided interval.
- Final-test target: approximately 100 accepted anchors if development throughput supports it.
- Horizon/action magnitude/angular half-width: H6 / 0.5 / 1.2 radians, matching the validated local setup.
- Candidate acceptance: exactly one contact transition and simulator-cost range at least 0.05.
- Readout candidates: the same endpoint pooled-latent ridge alpha grid only; no MLP.
- Search diagnostic: local scalar CEM, provisionally 5 iterations x 64 candidates, 8 elites; same seed schedule and budget for baseline and prospective repair.
- Dominance will be based on state-bootstrap mean recoverable regret headroom with a predeclared minimum material effect and separation rule, not whichever bar is numerically largest.

## Fresh development execution

- Split seed: 26,260,820. This range is disjoint from prior pilot/ranking states.
- Accepted 80 anchors from 147 attempts; rejected 35 without a contact boundary and 32 below the fixed 0.05 simulator-cost range. No multi-boundary state was accepted.
- Reference set: 81 actions per state; all 80 states have exactly one physical contact transition. Twelve of twelve exact cloned replays passed.
- Canonical BF16/batch-1 cache completed for 6,480 candidates. A duplicated shell wrapper was detected during generation; only the single active HDF5 writer was retained and generation resumed from its flushed anchor count without changing RNG state or samples.
- Five-fold anchor-grouped GT-only CV selected Ridge alpha 100 from the predeclared grid.
- GT/PRED OOF mean rho: 0.694/0.684. GT/PRED OOF regret: 0.0293/0.0330.
- Prediction gap: 0.0037, 95% state-bootstrap CI [-0.0127, 0.0191].
- Decision-metric gap: 0.0895 [0.0608, 0.1193].
- Baseline L2-CEM proposal gap: 0.0091 [0.0038, 0.0164].
- Baseline L2-CEM within-search selection gap: 0.1119 [0.0864, 0.1378]. This overlaps conceptually with the reference-set metric gap and is not added to it.
- Matched 65-query readout-CEM reduces development search regret by 0.0824 [0.0561, 0.1102]. This is development evidence only, not the prospective repair result.
- The frozen development diagnosis under the simple rule is `DECISION METRIC`.

## Final protocol freeze declaration

- Freeze alpha 100, endpoint pooled visual/proprio readout, 81-action H6 reference grid, `J_sim = 1-final_coverage`, 65-query scalar CEM, common random seeds, 5,000 state bootstraps, and the development dominance thresholds.
- Final test will contain 100 accepted anchors from split `bottleneck_test`, effective seed 28,260,820.
- Targeted repair, if the frozen final diagnosis is `DECISION METRIC`: replace L2 by the already-frozen readout inside the identical CEM.
- Wrong-layer control: rerun the identical official predictor in FP32 rather than BF16 while retaining the original L2 metric and identical CEM. This tests a cheap predictor-precision intervention without training or changing architecture.
- Targeted repair is prospectively successful only if mean search-regret improvement is at least 0.03 with bootstrap lower CI above zero. Actionability additionally requires at least a 2x improvement over the wrong-layer control; a wrong-layer effect of 0.02 or more is considered material.
- No final-test cache may exist before the freeze commit.

## Post-freeze execution notes

- Freeze commit: `6a49bdd802ccc8275a1a86fa7286a4b9c26362e0`.
- The first final-generation invocation failed before opening/creating the test cache because the reusable generator expected a development `split` key while the frozen protocol names the identical section `test`. This is a configuration-layout plumbing bug, not a scientific definition. The generator was changed to accept either key before retry; no protocol field or state was changed.

## Integrity boundary

All choices above are provisional until development is complete. The final protocol will be frozen and committed before any final-test state is generated. Diagnosis will be committed before repaired-search results are observed.

## Next automatic step

Implement reusable reference-set generation, decomposition metrics, and fully logged local CEM; add unit tests; then generate the new development split.
