# Full-Sequence Status

## 2026-08-20 — audit and design

- Starting commit verified: `8eb6279ca18e044bbe9070fd293c1fa2f7584044`.
- Push-T and Wall canonical manifests passed 57/57 SHA-256 and size checks.
- Paper-first milestone committed as `33df969`.
- Official Push-T configuration inspected: H6, 30 CEM iterations, 300 samples, 10 elites, six actions stepped, variance scale 1.0, terminal latent L2 with proprio weight 0.1.
- The model action dimension is expected to be 10 because each H6 step contains five normalized 2-D simulator controls; the complete one-shot sequence is therefore 60-D and executes 30 controls.
- The official offline Push-T dataset used by `goal_source: dset` is absent locally. The validation will preserve the checkpoint, preprocessing, action normalization, H6 rollout, CEM update, and query budget, while using fresh seed-generated actionable near-contact initial states and the canonical simulator target. This deviation will remain explicit.
- Current stage: traced-CEM and simulator replay feasibility benchmark.
- No final episodes have been generated or inspected.

## 2026-08-20 — feasibility benchmark

- Micro integration passed with 60-D sequences: 22.0 model candidates/s, 34.6 simulator sequences/s, and 196 terminal encodes/s.
- A literal 300-candidate FP32 model batch occupied about 15.8/16 GB VRAM and reached about 21.5 GB process private memory. It was terminated after 21 minutes without completing one 9000-query CEM trace. The stdout/stderr logs are retained; no scientific result artifact was produced.
- Decision recorded before further experiments: retain the official 300 candidates per CEM iteration but evaluate their scores in fixed groups of 75. Candidate samples, scores, elite selection, distribution updates, and total query budget are unchanged. This is an execution-only batching adaptation for the 16 GB GPU.
- Next automatic step: benchmark a complete 30×300 trace with query sub-batch 75, then run the pipeline smoke test.

## 2026-08-20 — development complete and final protocol frozen

- Full 75-sub-batch benchmark: CEM 302.99 s, simulator replay 270.88 s, GT endpoint encoding 22.31 s, peak allocated VRAM 4.53 GB.
- Traced CEM numerical equivalence tests: 2/2 passed.
- Five development baseline traces completed, each with 9001 simulator-evaluated sequences.
- Frozen readout: `StandardScaler + Ridge(alpha=0.1)`, SHA-256 `EDCFED77C89A0625C638FF45A6AC8A8BE9323AE946C5653DAEB6CDC4945EEACC`.
- Development readout diagnostics: GT rho 0.933/regret 0.086; predicted rho 0.801/regret 0.216; L2 regret 0.552.
- Two targeted integration checks were deliberately mixed: raw cost improvements +0.0265 and -0.1028. No scorer or protocol change was made in response.
- Final set fixed at 30 fresh episodes, seed range 20360830–20360859. No outcome filtering or replacement.
- Current stage: commit `configs/full_sequence/protocol.yaml` before any final state is generated.

## 2026-08-20 — frozen final run in progress

- Final protocol was committed before final generation as `3cbd3c50c472f1e3bcaad6e50c3b2835e007094c`.
- The immutable scientific-artifact audit passed 54/54 checks before launch.
- The final runner started at 2026-08-20 19:00:44 EDT with seeds 20360830–20360859 and no filtering, retry, or replacement.
- As of 2026-08-20 19:35 EDT, 2/30 complete JSON/NPZ episode pairs have been saved; episode 2 is running. The JSON is written after the NPZ and is therefore used as the completion marker.
- Only process health, artifact count/schema, GPU utilization, and stderr are being monitored. Final effect metrics remain uninspected until all 30 episodes are complete.
- Observed throughput is approximately 17 minutes per complete episode. No protocol, model, scorer, threshold, or statistic has changed after launch.
- Next automatic step: finish all 30 episodes, run the frozen analysis once, then generate plots and update the paper without retuning.

## 2026-08-20 — pre-unblinding analysis integrity audit

- Before inspecting any final metric, the analysis entry point was checked against the on-disk artifact schema.
- Engineering-only correction: analysis now requires both JSON and NPZ files and verifies the frozen episode/seed identity before computing statistics.
- Engineering-only correction: the phase manifest now includes raw final JSON/NPZ artifacts instead of hashing only configs, reports, results, and plots.
- The metric definitions, bootstrap, thresholds, interpretation rules, model, scorer, seeds, and candidate budget were not changed.

## 2026-08-20 — frozen final run checkpoint 5/30

- At 2026-08-20 20:27 EDT, 5/30 complete JSON/NPZ pairs were present and episode 5 had started automatically.
- All five completed episodes use the frozen contiguous seed order; no episode was retried, filtered, skipped, or replaced.
- The runner remained healthy with no stderr growth beyond the launch-time dependency warnings.
- CEM phases sustained approximately 99–100% GPU utilization. Observed peak temperature was 86°C with NVIDIA software thermal slowdown reported inactive, followed by rapid cooldown during simulator replay.
- Final metrics remain uninspected. The protocol and all scientific definitions remain frozen.

## 2026-08-20 — external GPU contention during episode 7

- At approximately 21:12 EDT, an unrelated user `cs2` process and Lossless Scaling began sharing the GPU. These processes were not launched, modified, or terminated by this experiment.
- Total device memory rose from roughly 6.6–7.0 GB to roughly 14.1 GB, leaving about 2 GB free. Device temperature reached 87°C and NVIDIA software thermal slowdown became active.
- The final runner remained alive and stderr showed no OOM or new error, but targeted-CEM wall-clock throughput slowed materially.
- Scientific computation is unchanged: the runner still uses the frozen candidates, query sub-batch 75, scorer, checkpoint, seeds, and budgets. If resource contention causes failure, the exact frozen command will be resumed from completed episode pairs without replacement.
- No final effect metric was inspected while diagnosing this resource event.

## 2026-08-20 — frozen final run checkpoint 10/30

- At 2026-08-20 22:22 EDT, 10/30 complete JSON/NPZ pairs were present and episode 10 had started automatically.
- External GPU contention had ended; device memory returned to roughly 6.6 GB, temperature was 72°C, and thermal slowdown was inactive.
- Runner stderr still contained only the launch-time dependency warnings. No final effect metric was inspected.
- Monitoring frequency was reduced at the user's request; the next routine checkpoint is 20/30, with immediate attention only if the runner exits or reports an error.

## 2026-08-21 — frozen final run checkpoint 20/30

- At 2026-08-21 01:32 EDT, 20/30 complete JSON/NPZ pairs were present and episode 20 had started automatically.
- GPU utilization was 99%, device memory roughly 6.5 GB, temperature 73°C, and thermal slowdown inactive.
- Runner stderr still contained only launch-time dependency warnings; no retry, skip, replacement, protocol change, or final-metric inspection occurred.
- Next routine checkpoint: 30/30 or immediate runner exit/error.

## 2026-08-21 — final generation complete; analysis serialization fix

- At 2026-08-21 04:21 EDT, all 30/30 JSON/NPZ pairs were complete and the runner exited normally. Stderr contained no runtime error.
- The first frozen-analysis invocation completed data loading/statistic construction but failed before writing or printing `summary.json` because YAML parsed `frozen_utc_date` as a Python `date`, which the standard JSON encoder cannot serialize.
- No result was inspected from the partially written CSV. The engineering-only fix adds `default=str` to summary JSON serialization; no metric, data row, bootstrap seed/sample count, threshold, or interpretation rule changed.
- Next step: rerun the identical frozen analysis command once, then inspect the resulting summary.

## 2026-08-21 — frozen analysis complete

- The identical frozen analysis command completed after the serialization-only repair.
- Frozen decision: `PREDICTION BECOMES IMPORTANT`.
- Prediction gap 0.1608 [0.0565, 0.2725]; metric gap 0.0950 [0.0026, 0.1978].
- Matched-budget metric repair changed regret 0.4974 → 0.4131; improvement 0.0843 [-0.00008, 0.1684], with 16/9/5 helped/unchanged/harmed.
- Strong metric-repair validation failed because the lower confidence bound crosses zero. No threshold, scorer, state, or protocol was changed after seeing this result.
- A separately labeled post-hoc failure audit was added for interpretation only. It retains episode 20 (repair -0.5956) and reports exploratory metric-gap/repair Spearman 0.792 [0.493, 0.955]. It is excluded from the frozen decision.
- Reports, workshop sources, plots, and machine-readable tables were updated without retuning.
- The final 30-episode schema/finite-value audit passed 30/30. Relevant tests passed 18/18.
- Frozen Push-T/Wall scientific artifacts passed the phase-end immutable audit; four explicitly authorized paper-source entries were excluded because this phase required updating them.
- The full-sequence manifest contains 116 files and passed independent size/SHA-256 verification.
- Current stage: complete; commit and hand off. No research job remains running.
