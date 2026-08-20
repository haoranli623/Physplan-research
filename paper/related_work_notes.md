# Related Work Notes

These are routing notes, not final prose. Links below were checked against primary paper/project pages on 2026-08-20.

## Latent world models for physical planning

- Terver et al., *What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?* ([arXiv 2512.24497](https://arxiv.org/abs/2512.24497)). This is the official JEPA-WMs paper/repository family underlying our checkpoint and evaluates architecture, objective, and planner design choices. Our distinction is state-level controlled substitution followed by prospective repair selection, not another architecture sweep.
- Zhou et al., *DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning* ([arXiv 2411.04983](https://arxiv.org/abs/2411.04983); [OpenReview](https://openreview.net/forum?id=D5RNACOZEI)). Establishes frozen DINOv2 patch features plus action-conditioned latent prediction and goal-feature planning.
- Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning* ([official Meta publication page](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)). Relevant action-conditioned latent world-model and robot-planning context.

## Learned planning metrics

- Park et al., *TLDR: Unsupervised Goal-Conditioned RL via Temporal Distance-Aware Representations* ([arXiv 2407.08464](https://arxiv.org/abs/2407.08464)). Use as evidence that temporally meaningful geometry is an established objective, not our novelty.
- Farahmand et al., *Value-Aware Loss Function for Model-based Reinforcement Learning* ([AISTATS/PMLR 2017](https://proceedings.mlr.press/v54/farahmand17a.html)). Formal precedent for fitting models according to downstream decision structure rather than generic predictive loss.
- Ayoub et al., *Model-Based Reinforcement Learning with Value-Targeted Regression* ([ICML/PMLR 2020](https://proceedings.mlr.press/v119/ayoub20a.html)). Another primary reference for decision-relevant model objectives.
- Our claim must not be “learned metric beats L2.” Distinction: controlled component attribution selects the repair before intervention.

## Model accuracy versus control utility

- Value-aware and decision-aware model learning already separate probabilistic/predictive fidelity from decision utility. Cite Farahmand et al. and Ayoub et al.; optionally add Voelcker et al., *Calibrated Value-Aware Model Learning with Probabilistic Environment Models* ([ICML/PMLR 2025](https://proceedings.mlr.press/v267/voelcker25a.html)) for calibration caveats.
- Distinction to establish: the present protocol estimates recoverable headroom at several interfaces using identical candidate sets and simulator oracles.

## Search/proposal diagnosis

- Rubinstein, *The Cross-Entropy Method for Combinatorial and Continuous Optimization* ([DOI](https://doi.org/10.1023/A:1010091220143)) is the primary CEM reference. We use finite candidates to expose coverage and selection separately; do not claim a new optimizer.
- Our audit retains every queried candidate and separately measures simulator-best queried cost versus the finite reference oracle.

## Diagnostic methodology

- Look for primary work on oracle substitutions, component-wise upper bounds, and prospective intervention validation.
- Avoid “first” claims. Novelty rests on the full diagnosis → committed repair prediction → independent intervention sequence.

## Citation verification queue

- Export exact BibTeX from arXiv/OpenReview/PMLR before TeX drafting.
- Check whether the workshop format permits citing 2026 contemporaneous work such as Temporal-Distance JEPA; do not rely on it for novelty positioning without reading the full primary manuscript.
