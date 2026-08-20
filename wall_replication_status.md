# Wall Replication Status

Last updated: 2026-08-20 15:43 EDT

## Current stage

P0-P3 are complete and Wall passed feasibility. D_wall_dev development and the frozen D_wall_diagnosis are complete. The diagnosis is `PREDICTION`. `wall_bottleneck_prediction.json` has been written while D_wall_repair remains unopened.

## Frozen diagnosis

- Diagnosis execution-code commit: `960bcd04ccf27f4eeab5893e4678db97c03155bf`.
- D_wall_diagnosis: 80 anchors, seeds starting at `1100000`, six diagnosis-only layouts.
- GT readout: rho `0.9428`, normalized regret `0.00359`.
- Prediction gap: `0.5636`, CI `[0.4871, 0.6373]`.
- Decision-metric gap: `-0.0279`, CI `[-0.0905, 0.0345]`.
- Search gap: `0.0333`, below the frozen material threshold of `0.05`.
- Frozen classification: **PREDICTION**.
- Repair cache was explicitly confirmed absent before writing the prediction file.

## Frozen Push-T boundary

- Canonical completion commit: `ad4a2524eb7b36ca59c56cf3478435d290334f2c`.
- Working tree was clean at phase start.
- All ten canonical files in `results/bottleneck_decomposition/artifact_manifest.json` match both recorded byte sizes and SHA256 hashes.
- Push-T configs, splits, scorer, results, plots, thresholds, and interpretations are read-only for this phase.
- Wall code and artifacts will live only under `wall_*`, `configs/wall_replication/`, `results/wall_replication/`, `plots/wall_replication/`, `docs/wall_replication.md`, and shared task-generic modules where necessary.

## Audited official Wall assets

- Official evaluation config exists at `configs/evals/simu_env_planning/wall/jepa-wm/wall_L2_cem_sourcerandstate_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml`.
- Official public checkpoint downloaded to `artifacts/checkpoints/jepa_wm_wall.pth.tar` (211,639,231 bytes; SHA256 `8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e`).
- Model family matches Push-T structurally: DINOv2 ViT-S/14 target encoder, six-layer AdaLN predictor, H6, five controls per latent step, two-dimensional actions.
- Wall state/proprio dimension is two. Official normalization statistics are present in `app/plan_common/datasets/__init__.py`.
- The simulator is the repository-native `DotWall`; action updates position by `2 * action` and the wall intersection logic clips crossings except through the door.
- The official checkpoint loaded at epoch 50 with all predictor and proprio-encoder keys matched. A BF16 H6 smoke prediction was finite; peak allocated VRAM was 269 MiB for two candidates.
- Exact-clone replay passed: independently constructed environments with different RNG seeds produced identical state trajectories (max difference 0) and byte-identical rendered observations for the same fixed layout/state/action sequence.
- Feasibility candidate geometry is a fixed 41-point grid of open-loop two-segment routes parameterized only by waypoint height. It uses 30 controls (H6 x frameskip 5). This choice and all feasibility thresholds were written to `configs/wall_replication/feasibility.yaml` before the ceiling result.
- Simulator cost is terminal Euclidean goal distance in pixels. In the first seed, candidates produced 38 distinct costs at 0.001-pixel resolution and a 22.71-pixel range; the best waypoint was at the door.

## Scientific guardrails

- Three-way split is mandatory after feasibility: Wall dev, diagnosis, and repair-validation states have disjoint seeds/states. Repair states cannot be opened before diagnosis and repair prediction are committed.
- Push-T-specific alpha, candidate angle slice, raw cost thresholds, horizon, cost normalization, and repair result do not transfer automatically.
- The same small action-blind readout family may be considered on Wall development, but its inputs and regularization must be selected without diagnosis/repair states.
- Headroom estimates remain non-additive controlled substitutions.
- A failed feasibility gate stops Wall; there will be no autonomous task switch.

## Feasibility questions and current status

- A. Official model/checkpoint: **pass**.
- B. Exact state/layout restoration: **pass**.
- C. Continuous simulator cost: **provisional pass; 30-state distribution pending**.
- D. Fixed informative reference set: **provisional pass; 30-state distribution pending**.
- E. GT-latent readout ceiling: **30-state feasibility run in progress**.
- F. Throughput/storage/VRAM: **one-anchor smoke pass**. Measured 67.1 simulator candidates/s, 62.5 encoded candidate trajectories/s, 30.7 predicted candidate trajectories/s, and 2.22 GiB peak allocated VRAM for 41 candidates.

## Feasibility result

- **PASS** under every threshold written before the 30-state run.
- Held-out GT readout: mean rho `0.8918`, top-5 retrieval `1.0`, mean normalized regret `2.30e-7`.
- Median cost range `27.1560` pixels; median 38 distinct costs among 41 candidates.
- Measured 30-state core runtime `64.44` seconds and peak allocated VRAM `2,602,539,008` bytes.
- Full evidence is in `WALL_FEASIBILITY_REPORT.md` and `results/wall_replication/feasibility.json`.

## Development-only evidence

- D_wall_dev: 60 anchors, seed range beginning `1000000`, six layouts. None overlap reserved diagnosis/repair seeds or layout tuples.
- Five-fold anchor-grouped selection chose ridge alpha `10.0` using GT-latent mean Spearman only.
- OOF GT readout: rho `0.9466`, normalized regret `0.0029`.
- OOF predicted-latent readout: rho `0.2209`, normalized regret `0.5738`.
- Development headroom: `G_pred=0.5709`; `G_metric=-0.0051`; `G_search=0.0494`.
- This is development evidence, not the Wall diagnosis.
- Minimal prediction repair was developed without diagnosis data: predictor-only fine-tuning under the original visual L2 plus 0.1 proprio L2. Epoch 5 was selected because held-out-dev latent loss fell from `0.9077` to `0.1507`.
- On the 12 held-out-dev anchors, matched 11-query normalized regret changed `0.5071 -> 0.0863` (improvement `0.4208`, CI `[0.1419, 0.6521]`). The wrong-layer metric readout improvement was `0.0489`, CI crossing zero. These values only justify freezing the repair mapping.

## Failures and debugging

- The base Anaconda interpreter lacked the required `torchvision`; the project-recorded `torch-gpu` environment is the valid environment and is now used explicitly.
- Windows loaded duplicate OpenMP runtimes when simulator and scientific packages were imported together. Runs set `KMP_DUPLICATE_LIB_OK=TRUE`, matching the previously documented local workaround; deterministic replay and exact render equivalence were verified after enabling it.
- Direct script invocation did not include the repository root on `sys.path`; reproducible commands use module invocation (`python -m scripts.wall_replication...`).

## Next action

Protocol freeze commit: `71e019f17df524602c9a5db89205da161a4e372a`.

Frozen protocol SHA256: `058852BEC36525F3ACB69F0E21D9F231F60A84427D5792ED3A2BE2A3E7BF0544`.

Commit `wall_bottleneck_prediction.json`. Only after that commit may D_wall_repair be generated.
