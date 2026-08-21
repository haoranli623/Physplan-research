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

The run commands and all interpretation rules were frozen in `configs/full_sequence/protocol.yaml` before final evaluation (protocol commit `3cbd3c5`). All 30 final episodes completed without filtering or replacement.

## Final result

- Frozen decision: `PREDICTION BECOMES IMPORTANT`.
- Prediction gap: 0.1608 [0.0565, 0.2725].
- Metric gap: 0.0950 [0.0026, 0.1978].
- Baseline/targeted adaptive regret: 0.4974/0.4131.
- Metric-repair improvement: 0.0843 [-0.00008, 0.1684]; helped/unchanged/harmed 16/9/5.

The local metric bottleneck therefore does not remain dominant under the official H6 planner. The metric repair is favorable on average but inconclusive under the frozen criterion. See `full_sequence_report.md` for the selection/coverage interaction and retained failure cases.

## Reproduction

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.run_final --config configs/full_sequence/protocol.yaml
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.analyze --config configs/full_sequence/protocol.yaml
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.plot_results
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.audit_failure_cases
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.verify_final_artifacts
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.build_manifest
D:\anaconda\envs\torch-gpu\python.exe -m scripts.full_sequence.verify_manifest
```

The runner skips only episodes with an existing completed JSON marker. Analysis requires all 30 paired JSON/NPZ files and validates each frozen episode/seed identity before producing results.
