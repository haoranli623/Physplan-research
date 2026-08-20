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
