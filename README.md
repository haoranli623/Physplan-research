# PhysPlan

**Controlled Bottleneck Attribution for World-Model Planning**

Accepted at NeurIPS 2026 Workshop on Physical Understanding (PhysUnderstand) — Poster.

PhysPlan studies where recoverable planning performance is lost in latent
world-model planning pipelines. Rather than assuming that poor planning
implies poor prediction, it measures recoverable headroom at prediction,
decision-metric, and search-related interfaces before selecting an
intervention.

PhysPlan builds on the official
[JEPA-WMs](https://github.com/facebookresearch/jepa-wms) implementation and
pretrained-model ecosystem. The upstream implementation, weights, datasets,
and original framework remain the work of the JEPA-WMs authors; the
PhysPlan-specific contributions are the bottleneck-attribution experiments,
analysis, intervention protocols, and research extensions in this repository.

## Core idea

A world-model planner can lose decision quality at several interfaces:

- **Prediction:** predicted latent futures differ from the corresponding true
  futures.
- **Decision metric / scoring:** the score ranks otherwise useful predicted
  futures poorly.
- **Search / proposal coverage:** the candidate set does not expose sufficiently
  good actions, or within-search selection fails to recover them.

PhysPlan changes one interface at a time, evaluates selected actions in the
simulator, and uses controlled oracle substitutions together with matched
interventions to estimate where recoverable decision headroom exists. These
gaps are diagnostic, paired headroom estimates. They are not assumed to be
additive and do not constitute a unique causal decomposition. The intended use
is practical: diagnosis should guide which component is repaired.

## Key frozen results

All values below are frozen. Regret is lower-is-better, and confidence
intervals use the independent evaluation state or episode as the resampling
unit.

### Controlled Local Push-T

- Prediction gap: `0.0071`, 95% CI `[-0.0051, 0.0198]`
- Decision-metric gap: `0.0851`, 95% CI `[0.0608, 0.1109]`
- Baseline BF16 + latent-L2 regret: `0.1237`
- BF16 + task-readout regret: `0.0421`
- Targeted improvement: `+0.0815`, 95% CI `[0.0563, 0.1079]`

Under this controlled local protocol, recoverable headroom is concentrated at
the decision metric rather than prediction. The repair result measures
**same-state intervention consistency**, not held-out validation, and this
protocol is not the official full Push-T planner.

### Wall

- Prediction gap: `0.5636`, 95% CI `[0.4871, 0.6373]`
- Metric gap: `-0.0279`, with a 95% CI that crosses zero
- Baseline regret: `0.5561`
- Predictor-repair regret: `0.1670`
- Predictor-repair improvement: `+0.3891`, 95% CI `[0.2930, 0.4804]`
- Wrong-layer metric-repair improvement: `+0.0069`, 95% CI
  `[-0.0489, 0.0625]`
- Full-trajectory latent MSE: `0.8564 → 0.1999`

Under the controlled Wall protocol, the diagnosis reverses: prediction has
substantial recoverable headroom, and the pre-specified predictor repair
recovers planning performance while the wrong-layer metric repair does not.
The prospective evaluation uses 80 states disjoint from development and
diagnosis. The predictor repair used one training seed, so state-level
confidence intervals are conditional on that fitted checkpoint and do not
measure training-seed variability.

### Official Push-T H6 Adaptive CEM

- Prediction gap: `0.1608`, 95% CI `[0.0565, 0.2725]`
- Metric gap: `0.0950`, 95% CI `[0.0026, 0.1978]`
- Scorer stress-test improvement: `+0.0843`, 95% CI
  `[-0.00008, 0.1684]`
- Episode outcomes: `16 improved / 9 unchanged / 5 worsened` across 30
  episodes

Both recoverable gaps are positive, with a larger prediction-gap point
estimate under the frozen classification rule. No paired confidence interval
was computed for prediction gap minus metric gap, so this does **not** establish
that prediction headroom is statistically larger. The scorer intervention is
a stress test rather than the diagnosis-selected predictor repair; its result
is directionally positive but statistically inconclusive. Because scoring
determines CEM elites and subsequent proposals, changing the scorer can also
change adaptive search coverage.

## Cross-regime takeaway

Across the studied protocols, the ordering of recoverable prediction and
decision-metric headroom changes. This change is descriptive rather than
causal. The protocols use regime-specific candidate sets and differ in task,
horizon, precision, action dimensionality, and planner/search procedure;
PhysPlan does not identify which individual factor causes the shift. Gap
magnitudes across regimes are therefore not directly comparable as a single
normalized physical quantity.

The strongest supported conclusion is methodological: diagnose recoverable
decision headroom before retraining the predictor, changing the decision
metric, or modifying search.

## Repository structure

- [`boundary_jepa/`](boundary_jepa/) — controlled Push-T diagnostics and the
  earlier feasibility-pilot implementation.
- [`wall_replication/`](wall_replication/) — Wall diagnosis and
  predictor-repair validation.
- [`scripts/`](scripts/) — experiment, analysis, plotting, and utility scripts.
- [`configs/`](configs/) — frozen protocols and experiment configurations.
- [`tests/`](tests/) — simulator, protocol, analysis, and artifact-integrity
  tests.
- [`results/`](results/) — compact frozen summaries, manifests, and selected
  reproducibility artifacts.
- [`plots/`](plots/) — research and publication figures.
- [`docs/`](docs/) — verified protocol notes and reproduction commands.
- [`paper/`](paper/) — manuscript source, figures, and tables.
- [`project_page/`](project_page/) — standalone static project-page source.
- [`app/`](app/), [`evals/`](evals/), and [`src/`](src/) — the upstream
  JEPA-WMs training, planning-evaluation, and model infrastructure on which
  PhysPlan builds.
- [`PhysPlan_Technical_Report.docx`](PhysPlan_Technical_Report.docx) — the
  completed-project technical report.

## Reproducibility notes

This GitHub repository is a versioned research snapshot. It retains the code,
frozen configurations, compact result summaries, manifests, tests, and selected
small reproducibility artifacts needed to audit the reported claims. Large
generated artifacts, model checkpoints, feature dumps, sweep outputs, and raw
rollout archives are intentionally excluded from Git and remain in separate
storage.

The README does not claim one-command reproduction. Follow the verified
commands and environment notes already recorded under [`docs/`](docs/) rather
than inferring new commands from this overview. The excluded large artifacts
are not needed to interpret the scientific claims reported above, although
some experiment reruns require the corresponding upstream checkpoints or
generated data.

## Research artifacts

- [PhysPlan technical report](PhysPlan_Technical_Report.docx)
- [Manuscript source, figures, and tables](paper/)
- [Static project-page source](project_page/)

## Upstream foundation: JEPA-WMs

PhysPlan is built on the official JEPA-WMs implementation and pretrained-model
ecosystem introduced by Basile Terver, Tsung-Yen Yang, Jean Ponce, Adrien
Bardes, and Yann LeCun in *What Drives Success in Physical Planning with
Joint-Embedding Predictive World Models?*

Upstream resources:

- [Official JEPA-WMs repository](https://github.com/facebookresearch/jepa-wms)
- [JEPA-WMs paper](https://arxiv.org/abs/2512.24497)
- [Official pretrained models](https://huggingface.co/facebook/jepa-wms)
- [Official hosted datasets](https://huggingface.co/datasets/facebook/jepa-wms)

The upstream world-model implementation, original experimental framework,
official pretrained weights, and hosted datasets are JEPA-WMs work and are not
claimed as PhysPlan contributions. PhysPlan contributes the controlled
interface-level attribution experiments, matched repair/control protocols,
analysis, and related extensions contained in this repository. Full upstream
model tables, installation instructions, and download links remain available
in the official JEPA-WMs repository and model hub.

Related upstream resources retained from the original project documentation:

- [DINO-WM paper](https://arxiv.org/abs/2411.04983) and
  [project page](https://dino-wm.github.io/)
- [V-JEPA 2 paper](https://arxiv.org/abs/2506.09985) and
  [repository](https://github.com/facebookresearch/vjepa2)
- [V-JEPA repository](https://github.com/facebookresearch/jepa)
- [DINOv3 repository](https://github.com/facebookresearch/dinov3)
- [DROID dataset](https://droid-dataset.github.io/droid/the-droid-dataset)

Please cite JEPA-WMs when using its implementation, weights, datasets, or
framework:

```bibtex
@misc{terver2025drivessuccessphysicalplanning,
      title={What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?},
      author={Basile Terver and Tsung-Yen Yang and Jean Ponce and Adrien Bardes and Yann LeCun},
      year={2025},
      eprint={2512.24497},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2512.24497}
}
```

## License

The upstream license and copyright attribution are preserved. See
[`LICENSE`](LICENSE) and [`THIRD-PARTY-LICENSES.md`](THIRD-PARTY-LICENSES.md)
for applicable terms and third-party components.
