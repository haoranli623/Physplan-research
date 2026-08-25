# PhysPlan Claim Audit

Audit completed after the first full LaTeX draft. The scientific source of truth is `PhysPlan_Technical_Report.docx`; saved artifacts are used only to verify reported values and protocol details. No paper-preparation experiment was run.

## Major quantitative and scientific claims

| Paper claim | Supporting report / artifact | Exact frozen evidence | Statistical status | Allowed wording and limitation |
|---|---|---|---|---|
| PhysPlan estimates recoverable headroom through controlled oracle substitutions. | Report Sec. 3; `paper/EVIDENCE_MAP.md` | GT-readout, predicted-readout, predicted-L2, queried-oracle, and reference-oracle conditions | Methodological definition | “Estimates recoverable headroom under the specified substitution.” Never “uniquely assigns causal responsibility” or “additive decomposition.” |
| Initial state or episode is the independent unit. | Report Secs. 3.3, 4–6 | State/episode paired bootstrap; candidate actions are not independent trials | Protocol fact | State-level/episode-level inference only. |
| Controlled local Push-T is primarily decision-metric limited. | Report Sec. 4; `results/bottleneck_decomposition/test/decomposition_summary.json` | GT readout rho/regret 0.696/0.0289; predicted readout 0.689/0.0360; prediction gap 0.0071, CI [-0.0051, 0.0198]; metric gap 0.0851, CI [0.0608, 0.1109]; proposal gap 0.0074 | Metric gap clears zero; prediction gap does not | Always call this the “controlled local Push-T protocol/action family,” not the official Push-T planner. No universal metric-bottleneck claim. |
| Local metric repair recovers the diagnosed headroom. | Report Sec. 4.6; `results/bottleneck_decomposition/repair/repair_summary.json` | regret 0.1237 to 0.0421; improvement +0.0815, CI [0.0563, 0.1079] | CI excludes zero | “Diagnosis-selected metric repair improved mean normalized regret under the controlled local protocol.” |
| Local repair has heterogeneous effects. | Same source | 62% improved, 21% unchanged, 17% harmed | Descriptive state accounting | Do not imply every state benefits. |
| Increasing predictor inference precision did not recover local planning performance. | Report Sec. 4.6; local repair summary | FP32 predictor + latent-L2 improvement +0.0017; CI crosses zero | Control is null/inconclusive | Exact allowed meaning: numerical-precision control only. Never “improving the predictor did not help.” |
| Independent Wall replication is prediction limited. | Report Sec. 5; `results/wall_replication/diagnosis/diagnosis.json` | prediction gap 0.5636, CI [0.4871, 0.6373]; metric gap -0.0279, CI crosses zero; proposal gap 0.0333; 60/80/80 disjoint development/diagnosis/repair states | Prediction gap strongly positive; metric gap not distinguishable from zero | Scope to controlled Wall waypoint protocol. “Independent” refers to disjoint pre-registered states/protocol, not another laboratory. |
| Wall predictor repair succeeds on unseen states and wrong-layer metric repair does not. | Report Sec. 5.5; `results/wall_replication/repair/repair.json` | baseline regret 0.5561; repaired 0.1670; improvement 0.3891, CI [0.2930, 0.4804]; wrong-layer improvement 0.0069, CI [-0.0489, 0.0625] | Targeted CI excludes zero; control CI crosses zero | Mention evaluation on 80 entirely new repair states and one training seed. |
| Wall repair lowers average prediction MSE but MSE does not guarantee state-wise control gain. | Same source and report Sec. 5.6 | full-trajectory latent MSE 0.8564 to 0.1999; anchor 31 regret 0.215 to 0.991 despite lower MSE | Aggregate mechanism check plus retained counterexample | “Lower average prediction error is not sufficient to guarantee state-wise decision improvement.” No claim that MSE is generally irrelevant. |
| Official Push-T H6 has mixed headroom with prediction greater than metric. | Report Sec. 6; `results/full_sequence/summary.json` | GT/pred-readout/pred-L2 regret 0.1724/0.3332/0.4282; prediction gap 0.1608, CI [0.0565, 0.2725]; metric gap 0.0950, CI [0.0026, 0.1978] | Both CIs positive; frozen rule classifies prediction as larger | “Mixed, with prediction headroom > metric headroom” and “bottleneck migration.” Do not say metric is irrelevant. |
| H6 scorer repair is directionally positive but inconclusive. | Report Sec. 6.6; H6 summary | baseline/repair regret 0.4974/0.4131; improvement +0.0843, CI [-0.00008, 0.1684]; 16/9/5 improved/unchanged/worsened over 30 episodes | CI narrowly crosses zero | Required wording: “directionally positive but statistically inconclusive.” Never “significant” or “validated successful repair.” |
| A scorer changes adaptive CEM search coverage, so fixed-trace attribution differs from end-to-end intervention. | Report Secs. 6.3–6.7; H6 raw traces and summary | baseline/repair selection gap 0.1834 to 0.0983; coverage gap 0.0297 to 0.0671; retained episode worsens 0.4044 to 1.0000 with coverage gap 0.5811 | Mechanistic descriptive audit | State that score changes elite selection and future proposals. Trace union is a finite audit reference, not a global oracle. |
| Dominant bottleneck changes across planning regimes. | Report Secs. 7–8; all three frozen summaries | Local: metric > prediction; Wall: prediction >> metric; H6: both positive, prediction > metric | Cross-regime empirical synthesis | Scope to the three studied protocols. Never generalize broadly across robotics or real robots. |
| Reproducibility record is complete for the frozen project. | Report Sec. 11; repository manifests/tests | 30/30 H6 JSON/NPZ pairs; 18/18 related tests; 53/53 immutable historical artifacts; 117/117 SHA-256 checks; final commit `ceb6122`; protocol commit `3cbd3c5` | Repository verification metadata | Commit hashes may appear anonymously; no identifying repository URL. |

## Related-work verification

Every bibliography entry used in the manuscript was checked against a primary paper page, official proceedings record, DOI record, or official arXiv record. The bounded set positions latent world-model planning, CEM, and decision-aware model objectives; none is used to modify PhysPlan results. Key primary records include PMLR for PlaNet, DINO-WM, value-aware learning, value-targeted regression, calibrated value-aware learning, and Push-T; official NeurIPS/ICLR proceedings for PETS, Value Equivalence, and TD-MPC2; the original CEM DOI; and official arXiv records for JEPA-WMs and V-JEPA 2.

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
| “This is merely an ablation study.” | Secs. 3 and 6 tie each candidate-matched contrast to a prospective intervention and wrong-layer control. | Attribution remains intervention-specific rather than uniquely causal. |
| “Are oracle gaps additive?” | Secs. 3 and 6 state explicitly that they overlap and must not be summed. | Interface interactions remain. |
| “Why not simply improve prediction?” | Local Push-T shows negligible prediction headroom; Wall and H6 show when prediction is important. | FP32 is only a numerical-precision control locally. |
| “Does local Push-T generalize?” | Wall reverses the diagnosis; H6 changes it again. | Only two simulation tasks and one world-model family are studied. |
| “Why does H6 differ?” | Secs. 3.4 and 5.3 explain longer 60-D sequences, accumulated error, and adaptive proposal updates. | No new predictor repair was trained for H6. |
| “Is the H6 repair significant?” | Abstract, Results, table, figure, and Limitations all label it inconclusive. | Only 30 episodes. |
| “Does scoring confound search in CEM?” | The method diagram and Sec. 3.4 distinguish frozen-trace ranking from end-to-end adaptive search. | Trace union is not a global oracle. |
| “Is Wall validation independent?” | Sec. 4.2 specifies disjoint 60/80/80 development, diagnosis, and repair states and prospective freezing. | The predictor repair has one training seed. |
| “Does lower prediction MSE imply better control?” | The retained Wall counterexample shows it does not guarantee state-wise improvement. | The aggregate Wall repair remains positive. |
| “What is the methodological contribution?” | Candidate-matched interface substitutions plus prospective diagnosis-selected repair across different failure regimes. | No claim of inventing world models, CEM, or task-aware scores. |
