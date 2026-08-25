# PhysPlan Claim Audit

Audit refreshed after the reviewer-driven revision. Frozen machine-readable artifacts, protocol configs, and evaluation/training code were checked first; the technical report was used as a narrative cross-check. No paper-preparation experiment was run, and no frozen value changed.

## Major quantitative and scientific claims

| Paper claim | Supporting report / artifact | Exact frozen evidence | Statistical status | Allowed wording and limitation |
|---|---|---|---|---|
| PhysPlan estimates recoverable headroom through controlled oracle substitutions. | Report Sec. 3; `paper/EVIDENCE_MAP.md` | GT-readout, predicted-readout, predicted-L2, queried-oracle, and reference-oracle conditions | Methodological definition | “Estimates recoverable headroom under the specified substitution.” Never “uniquely assigns causal responsibility” or “additive decomposition.” |
| Initial state or episode is the independent unit. | Report Secs. 3.3, 4–6 | State/episode paired bootstrap; candidate actions are not independent trials | Protocol fact | State-level/episode-level inference only. |
| Controlled local Push-T is primarily decision-metric limited. | Report Sec. 4; `results/bottleneck_decomposition/test/decomposition_summary.json` | GT readout rho/regret 0.696/0.0289; predicted readout 0.689/0.0360; prediction gap 0.0071, CI [-0.0051, 0.0198]; metric gap 0.0851, CI [0.0608, 0.1109]; proposal gap 0.0074 | Metric gap clears zero; prediction gap does not | Always call this the “controlled local Push-T protocol/action family,” not the official Push-T planner. No universal metric-bottleneck claim. |
| Local metric repair recovers the diagnosed headroom. | Report Sec. 4.6; `results/bottleneck_decomposition/repair/repair_summary.json` | regret 0.1237 to 0.0421; improvement +0.0815, CI [0.0563, 0.1079] | CI excludes zero | “Diagnosis-selected metric repair improved mean normalized regret under the controlled local protocol.” |
| Local repair has heterogeneous effects. | Same source | 62% improved, 21% unchanged, 17% harmed | Descriptive state accounting | Do not imply every state benefits. |
| Increasing predictor inference precision did not recover local planning performance. | Report Sec. 4.6; local repair summary | FP32 predictor + latent-L2 improvement +0.0017; CI crosses zero | Control is null/inconclusive | Exact allowed meaning: numerical-precision control only. Never “improving the predictor did not help.” |
| Disjoint-state Wall validation is prediction limited under its frozen rule. | `results/wall_replication/diagnosis/diagnosis.json`; `configs/wall_replication/protocol.yaml`; diagnosis code | prediction gap 0.5636, CI [0.4871, 0.6373]; metric gap -0.0279, CI [-0.0905, 0.0345]; proposal gap 0.0333; 60/80/80 disjoint development/diagnosis/repair states | Prediction gap strongly positive; metric gap not distinguishable from zero | Scope to controlled Wall waypoint protocol. Call it prospective disjoint-state validation, not external replication. |
| Wall predictor repair succeeds on unseen states and wrong-layer metric repair does not. | `results/wall_replication/repair/repair.json`; `configs/wall_replication/repair.yaml`; `scripts/wall_replication/train_predictor_repair.py` | baseline regret 0.5561; repaired 0.1670; improvement 0.3891, CI [0.2930, 0.4804]; wrong-layer improvement 0.0069, CI [-0.0489, 0.0625] | Targeted state-bootstrap CI excludes zero; control CI crosses zero | Evaluation uses 80 entirely new repair states. One training seed means the CI is conditional on one fitted checkpoint and excludes training variability. |
| Wall repair lowers average prediction MSE but MSE does not guarantee state-wise control gain. | Same source and report Sec. 5.6 | full-trajectory latent MSE 0.8564 to 0.1999; anchor 31 regret 0.215 to 0.991 despite lower MSE | Aggregate mechanism check plus retained counterexample | “Lower average prediction error is not sufficient to guarantee state-wise decision improvement.” No claim that MSE is generally irrelevant. |
| Official Push-T H6 has mixed headroom and meets the frozen prediction-important rule. | `results/full_sequence/summary.json`; `configs/full_sequence/protocol.yaml`; `scripts/full_sequence/analyze.py` | GT/pred-readout/pred-L2 regret 0.1724/0.3332/0.4282; prediction gap 0.1608, CI [0.0565, 0.2725]; metric gap 0.0950, CI [0.0026, 0.1978] | Both CIs positive; prediction point estimate is larger and passes the frozen rule; no paired CI was computed for their difference | Say “larger point estimate under the frozen rule,” not that prediction statistically exceeds metric. The H6 scorer test is not a diagnosis-selected predictor repair. |
| H6 scorer repair is directionally positive but inconclusive. | Report Sec. 6.6; H6 summary | baseline/repair regret 0.4974/0.4131; improvement +0.0843, CI [-0.00008, 0.1684]; 16/9/5 improved/unchanged/worsened over 30 episodes | CI narrowly crosses zero | Required wording: “directionally positive but statistically inconclusive.” Never “significant” or “validated successful repair.” |
| A scorer changes adaptive CEM search coverage, so fixed-trace attribution differs from end-to-end intervention. | Report Secs. 6.3–6.7; H6 raw traces and summary | baseline/repair selection gap 0.1834 to 0.0983; coverage gap 0.0297 to 0.0671; retained episode worsens 0.4044 to 1.0000 with coverage gap 0.5811 | Mechanistic descriptive audit | State that score changes elite selection and future proposals. Trace union is a finite audit reference, not a global oracle. |
| Recoverable-headroom ordering changes across planning protocols. | All three frozen summaries and configs | Local: metric diagnosis; Wall: prediction diagnosis; H6: both positive, prediction-important under frozen rule | Descriptive cross-protocol synthesis | “Migration” is descriptive. Task, candidate family, horizon, precision, and planner differ together, so the study does not identify the causal source of the change. |
| Reproducibility record is complete for the frozen project. | Report Sec. 11; repository manifests/tests | 30/30 H6 JSON/NPZ pairs; 18/18 related tests; 53/53 immutable historical artifacts; 117/117 SHA-256 checks; final commit `ceb6122`; protocol commit `3cbd3c5` | Repository verification metadata | Commit hashes may appear anonymously; no identifying repository URL. |

## Related-work verification

Every bibliography entry used in the manuscript was checked against a primary paper page, official proceedings record, DOI record, or official arXiv record. The revised positioning explicitly includes Objective Mismatch and planning-aware prediction evaluation; neither is used to alter PhysPlan results. PhysPlan does not claim oracle substitution or task-aware scoring as individually new. Its defensible novelty is the candidate-matched, precommitted diagnosis--repair--wrong-layer-control workflow and the fixed-trace/adaptive-search distinction.

## Unsupported-statement check

No unsupported major quantitative statement remains in the manuscript. The following tempting claims were explicitly excluded:

- that world-model planners are universally metric limited or prediction limited;
- that the gaps are additive or uniquely causal;
- that FP32 tests a general predictor improvement;
- that the H6 scorer repair is statistically significant;
- that controlled local Push-T is the official planner;
- that PhysPlan has broad real-robot, VLA, or learned-policy validation.

## Reviewer attack pass

| Likely criticism | Manuscript response | Remaining limitation |
|---|---|---|
| “This is merely an ablation study.” | Related Work and Secs. 3 and 6 concede that oracle replacement is familiar, then identify the added prospective workflow, candidate matching, state-level inference, and wrong-layer controls. | Methodological novelty is organizational and diagnostic, not a new oracle operator. |
| “Are oracle gaps additive?” | Secs. 3 and 6 state explicitly that they overlap and must not be summed. | Interface interactions remain. |
| “Why not simply improve prediction?” | Local Push-T shows negligible prediction headroom; Wall and H6 show when prediction is important. | FP32 is only a numerical-precision control locally. |
| “Does local Push-T generalize?” | Wall reverses the diagnosis; H6 changes it again. | Only two simulation tasks and one world-model family are studied. |
| “Why does H6 differ?” | Secs. 3.4 and 5.3 establish that the protocol changes and explain adaptive proposal updates. | Task/planner factors are confounded; no causal mechanism is isolated and no H6 predictor repair was trained. |
| “Is the H6 repair significant?” | Abstract, Results, table, figure, and Limitations all label it inconclusive. | Only 30 episodes. |
| “Does scoring confound search in CEM?” | The method diagram and Sec. 3.4 distinguish frozen-trace ranking from end-to-end adaptive search. | Trace union is not a global oracle. |
| “Is Wall validation independent?” | The paper now calls it disjoint-state prospective validation and specifies 60/80/80 development, diagnosis, and repair states. | It is not an external replication; the predictor repair has one training seed and its CI is conditional on that checkpoint. |
| “Does lower prediction MSE imply better control?” | The retained Wall counterexample shows it does not guarantee state-wise improvement. | The aggregate Wall repair remains positive. |
| “What is the methodological contribution?” | Candidate-matched interface substitutions plus prospective diagnosis-selected repair across different failure regimes. | No claim of inventing world models, CEM, or task-aware scores. |
