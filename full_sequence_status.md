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
