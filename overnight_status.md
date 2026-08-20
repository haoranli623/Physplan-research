# Boundary-JEPA Overnight Status

Last updated: 2026-08-20 02:39 EDT

## Current stage

P1 — parallel state-disjoint cloned-sweep generation; evaluation feature/probe pipeline ready for the completed cache.

## Initial state

- Workspace `D:\projects\physplan` was empty and was not a Git repository.
- Initialized from the official public repository `https://github.com/facebookresearch/jepa-wms.git`.
- Upstream commit: `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.
- Working branch: `boundary-jepa-overnight`.
- No pre-existing experiment outputs or user changes existed in the workspace.
- Machine: Windows, RTX 3080 Laptop GPU 16 GB, Ryzen 9 5900HX (8C/16T), 32 GB RAM.
- Free space at start: D: ~264 GiB.
- Candidate interpreter: `D:\anaconda\envs\torch-gpu\python.exe`, Python 3.11.15, PyTorch 2.11.0+cu128, CUDA available.
- Official project requests Python >=3.10,<3.11. The existing CUDA environment is Python 3.11; this compatibility deviation will be smoke-tested and documented rather than silently assumed.

## Locked scientific protocol

- Primary environment: Push-T.
- Physical labels remain separate from task outcome.
- Probe is action-blind and trained only on ground-truth transition/trajectory latents, then frozen.
- Primary boundary statistic uses cloned anchor states; nearby actions are not independent samples.
- Model and oracle receive exactly the same candidate action set for local regret.
- No A/B/C/D training until the feasibility pilot is at least a credible WEAK GO.

## Concrete overnight execution plan

1. Inspect official Push-T simulator, model wrapper, checkpoint loader, preprocessing, configs, and existing tests.
2. Install only the minimal official Push-T/model dependencies; download the official Push-T JEPA-WM checkpoint and DINOv2 encoder through official loaders.
3. Add deterministic clone/replay tests and a minimal rendered observation/model-forward smoke test.
4. Define stable physical event extraction from simulator contact information; manually/statistically validate before generation.
5. Implement resumable cloned-state sweep generation with state-disjoint probe/evaluation splits and raw auditable records.
6. Cache frozen true/predicted latent trajectories; train and freeze the action-blind transition probe.
7. Compute three maps/boundaries: simulator truth, probe-on-GT-latents, probe-on-predicted-latents.
8. Compute per-anchor latent error, boundary error, and matched-candidate local planning regret.
9. Bootstrap state-level correlations; generate non-cherry-picked plots and `pilot_report.md`.
10. If GO/credible WEAK GO, scale data and begin the highest-value matched intervention run; otherwise run only protocol-preserving diagnostics.

## Completed work

- Verified the workspace was empty.
- Cloned and inspected the official repository, README, package metadata, Push-T dataset/environment entry points, evaluation documentation, Git history/status, and machine resources.
- Created the dedicated branch and recorded upstream state.
- Installed the minimal Windows-compatible Push-T/model/scientific dependencies into the existing CUDA environment.
- Downloaded the official 211,639,615-byte Push-T JEPA-WM checkpoint and official DINOv2 ViT-S/14 weights.
- Validated exact cloned replay: maximum state error 0, maximum RGB error 0, identical contact sequence.
- Loaded the official epoch-50 predictor/proprio checkpoint with all keys matched and ran a real simulator-to-latent predictor smoke test.
- Confirmed the official temporal interface: five 2-D control actions are normalized and concatenated into each 10-D latent-step action.
- Added tested, resumable HDF5 sweep generation; a 5-anchor smoke run required 6 attempts and produced clean single boundaries.
- Added batched frozen encoder/predictor feature caching, action-blind current/endpoint/trajectory probes, calibrated metrics, boundary/regret analysis, state bootstrap, deterministic non-cherry-picked plotting, and report generation.
- Six focused Boundary-JEPA tests currently pass.

## Experiment currently running

- Two CPU processes are generating disjoint caches in parallel: 240 probe-training anchors and 100 held-out evaluation anchors, each with 41 actions and two latent steps. At 02:39 the caches were 57/240 and 97/100 complete.

## Failures/debugging notes

- The existing `torch-gpu` environment has core PyTorch/CUDA but lacks the simulator, plotting, scientific, Hydra, and Hugging Face dependencies.
- Repository metadata targets Python 3.10. A dedicated Python 3.10 environment may be required if the minimal Python 3.11 smoke test exposes incompatibility.
- Direct script invocation does not put the repository root on `sys.path`; reproducible commands use `python -m scripts.boundary_jepa.<name>`.
- Gym emits its expected NumPy-2 compatibility warning, but deterministic simulator tests and generation succeed on the exercised API.

## Important decisions

- Use the official Push-T JEPA-WM checkpoint, not an unofficial reimplementation.
- Avoid the full dependency set initially because PointMaze/D4RL and RoboCasa are irrelevant to P0–P2 and add Windows-specific failure risk.
- Do not download the offline Push-T training dataset until it is shown to be required for anchor selection or matched intervention training; simulator-generated pilot data comes first.
- Primary pilot physical label is only simulator-native any-contact over the ten-control-step rollout. Persistent/lost-contact are saved but not used in the primary pilot.
- Clean one-dimensional boundaries are selected by a predeclared simulator-only rule: a one-sided angular slice must contain exactly one contact/no-contact crossing.
- Canonical planner goal is fixed in config; all candidates are ranked with the same official terminal latent-L2 objective plus the official 0.1 proprio weight.
- Pilot GO/NO-GO thresholds were encoded before probe/evaluation results were available.

## Next automatic step

As soon as evaluation generation closes its HDF5 file, cache GT/predicted features on the GPU; continue probe-train generation concurrently, then encode it, train/freeze the probe, and run the full 100-state analysis.
