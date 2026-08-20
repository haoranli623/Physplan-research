# Wall Replication Summary

The independent Wall replication succeeded and identified a different limiting component from Push-T.

On 80 frozen diagnosis anchors, the action-blind GT-latent readout had mean Spearman `0.943` and normalized regret `0.0036`. Replacing GT futures with official predicted futures produced prediction headroom `0.5636` (95% CI `[0.4871, 0.6373]`), while decision-metric headroom was `-0.0279` with a CI spanning zero. The precommitted rule therefore diagnosed **PREDICTION**, not decision metric.

The diagnosis and repair prediction were committed in `8559594` before generating any of the 80 disjoint repair anchors. On that unseen set, the fixed predictor-only repair reduced matched 11-query normalized regret from `0.5561` to `0.1670`, an improvement of `0.3891` `[0.2930, 0.4804]`. The preregistered wrong-layer metric readout improved by only `0.0069` `[-0.0489, 0.0625]`. The targeted-minus-control contrast was `0.3822` `[0.2797, 0.4798]`.

The intervention helped 62/80 states, was unchanged on 7/80, and harmed 11/80. It also reduced a secondary full-trajectory visual latent MSE from `0.8564` to `0.1999`; this MSE diagnostic was not part of the repair success rule.

What this supports: a fixed controlled-attribution protocol distinguished a Push-T metric bottleneck from a Wall prediction bottleneck and prospectively selected the higher-value intervention on fully unseen Wall states.

What this does not support: universal bottleneck identification, official full-CEM validity, or a claim that lower prediction loss always improves decisions. The Wall experiment is still a controlled one-dimensional waypoint candidate study, and 13.75% of repair states were harmed.
