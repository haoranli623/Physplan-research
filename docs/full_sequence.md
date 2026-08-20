# Official Push-T H6 Full-Sequence Validation

This validation asks whether the Push-T local metric-bottleneck diagnosis survives the official 30-iteration, 300-sample, 10-elite H6 CEM regime.

The checkpoint, preprocessing, action normalization, 6×10 model-action representation, terminal L2 objective, proprio weight, and CEM update match the official configuration. Each model action contains five consecutive normalized two-dimensional controls, so one plan contains 30 simulator controls and 60 scalar action variables.

To avoid the severe throughput collapse observed when 300 sequences occupy nearly all laptop VRAM, each sampled 300-candidate CEM population is scored in fixed sub-batches. This changes neither candidates, scores, elite selection, nor query count; it is an execution-only batching adaptation. Its size is fixed from development throughput before final evaluation.

The official offline Push-T dataset is not present on this machine. Consequently, `goal_source: dset` cannot be reproduced exactly. The experiment uses fresh, seed-determined actionable near-contact states and the canonical simulator target. No state is accepted or rejected based on planning outcome.

Development episodes are used only to test integration and fit one action-blind `StandardScaler + Ridge` terminal-latent task readout. Ridge alpha is selected by grouped episode cross-validation. The readout receives pooled terminal visual and proprio latent features and never receives action.

Final attribution uses one baseline CEM trace per episode. On exactly the same retained sequences it compares:

- encoded simulator endpoint + frozen readout;
- predicted endpoint + the same frozen readout;
- predicted endpoint + official latent L2.

The final candidate after the last CEM update (the official returned mean sequence) is appended to the 9000 queried candidates and simulator-evaluated. Separately, a matched-budget targeted CEM uses the frozen readout; only the scoring function changes. Its adaptive candidate trace is not treated as matched to the baseline trace.

Every final candidate sequence, score, simulator cost, distribution mean/std, and final state is retained. Episode is the independent statistical unit.

Run commands will be frozen in `configs/full_sequence/protocol.yaml` after development and before final evaluation.
