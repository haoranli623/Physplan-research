# Wall Feasibility Report

## Decision

**PASS.** All predeclared checks in `configs/wall_replication/feasibility.yaml` passed. This authorizes Wall protocol development, but is not a bottleneck result.

## Official baseline

- Official public Wall JEPA-WM checkpoint: 211,639,231 bytes, SHA256 `8efb0623cfba1cb3ca210de26f7579c83dd24936635f11989c515afcb23bea1e`.
- Checkpoint epoch: 50. Predictor and proprio encoder reported all keys matched.
- Architecture/config: frozen DINOv2 ViT-S/14 encoder, six-layer AdaLN action-conditioned predictor, H6, frameskip 5, two-dimensional action and proprioception.
- BF16 predictions and true latents were finite in the smoke test.

## State restoration

Independent `DotWall` instances with different RNG seeds produced exactly the same rollout from the same fixed layout, state, and action sequence:

- maximum absolute state difference: `0.0`
- rendered observations: byte-identical
- vectorized rollout/render implementation: state-identical and image-identical to direct environment stepping

## Cost and candidates

The simulator cost is terminal Euclidean distance to the goal in the native 65-pixel workspace. The fixed reference set contains 41 open-loop H6 action sequences. Each sequence follows a two-segment route through one waypoint on the wall plane; only waypoint height changes.

Across 30 feasibility-only anchors:

- median candidate cost range: `27.1560` pixels
- median distinct costs after 0.001-pixel rounding: `38 / 41`
- candidates with any collision: `92.68%`
- candidates ending on the goal side of the wall: `41.06%`

This is a continuous, geometrically meaningful ranking problem rather than a success-only label.

## GT-latent ceiling

A small action-blind ridge readout was fit on 20 feasibility anchors and evaluated on 10 disjoint held-out anchors. Its inputs were signed, absolute, and squared terminal-to-goal differences of pooled true visual and proprio latents. Ridge alpha was chosen by five-fold anchor-grouped validation using fit anchors only.

- selected alpha: `0.1`
- held-out mean Spearman: `0.8918`
- held-out median Spearman: `0.9755`
- held-out top-5 retrieval: `1.000`
- held-out mean normalized selection regret: `2.30e-7`

One held-out state had rho `0.3575`; it was retained. The ceiling is therefore strong in aggregate but not uniformly strong.

For context only, on these same held-out feasibility states:

- GT latent L2 mean rho: `0.7663`
- predicted latent L2 mean rho: `0.6815`
- predicted latent L2 mean selection regret: `3.1206` pixels

These feasibility observations are not a frozen diagnosis and will not be used as Wall diagnosis evidence.

## Throughput and resources

Measured for 30 anchors / 1,230 candidates:

- exact simulator: `61.83` candidates/s
- true trajectory encoder: `158.95` candidate trajectories/s
- H6 predictor: `35.98` candidate trajectories/s
- peak allocated VRAM: `2,602,539,008` bytes
- compressed cache: `21,375,095` bytes
- estimated 120-anchor core pipeline time: `0.0716` hours

The experiment is practical on the available RTX 3080 16 GB. Increasing memory occupancy is not justified by this workload.

## Known engineering notes

- The correct project interpreter is `D:\anaconda\envs\torch-gpu\python.exe`; the base interpreter lacks torchvision.
- The Windows environment requires `KMP_DUPLICATE_LIB_OK=TRUE` because two OpenMP runtimes are loaded. Exact replay and rendering were explicitly tested under this setting.
- Gym emits an upstream maintenance warning; it does not invalidate the deterministic native `DotWall` checks.

## Next step

Use only `D_wall_dev` to fix the Wall readout alpha, query subset, normalization, bootstrap, dominance rule, and repair mapping. Do not generate diagnosis or repair states until the protocol is frozen and committed.
