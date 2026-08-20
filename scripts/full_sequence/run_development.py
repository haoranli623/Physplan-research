"""Generate fresh full-sequence development traces and freeze one ridge readout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from boundary_jepa.baseline import load_official_pusht_jepa
from boundary_jepa.full_sequence import (
    FeatureCapturingScorer,
    TorchReadout,
    TracedCEMPlanner,
    denormalize_sequences,
    encode_endpoint_batch,
    encode_goal,
    encode_single_state,
    sample_actionable_state,
    simulate_sequences,
)


def _planner(cfg: dict, bundle, seed: int) -> TracedCEMPlanner:
    p = cfg["planner"]
    return TracedCEMPlanner(
        iterations=p["iterations"], num_samples=p["samples"], num_elites=p["elites"],
        horizon=cfg["model"]["horizon"], action_dim=bundle.model.action_dim,
        var_scale=p["var_scale"], momentum_mean=p["momentum_mean"],
        momentum_std=p["momentum_std"], device=bundle.model.device, seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_sequence/development.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg["outputs"]["directory"])
    episode_dir = Path(cfg["outputs"]["episode_directory"])
    out.mkdir(parents=True, exist_ok=True)
    episode_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_official_pusht_jepa(cfg["checkpoint"], device=cfg["device"])
    if int(bundle.model.action_dim) != int(cfg["model"]["model_action_dim"]):
        raise RuntimeError("Official model action dimension changed")
    z_goal = encode_goal(bundle)
    all_true, all_pred, all_cost, all_l2, all_group = [], [], [], [], []
    timings = []
    for episode in range(int(cfg["episodes"])):
        path = episode_dir / f"episode_{episode:03d}.npz"
        episode_seed = int(cfg["seed"]) + episode
        if not path.exists():
            state = sample_actionable_state(episode_seed)
            z0 = encode_single_state(bundle, state)
            scorer = FeatureCapturingScorer(
                bundle, z_goal, alpha=float(cfg["planner"]["terminal_proprio_alpha"]),
                model_query_batch_size=int(cfg["model"]["model_query_batch_size"]),
            )
            trace = _planner(cfg, bundle, episode_seed).plan(z0, scorer)
            normalized = np.concatenate([trace.flat_candidates, trace.selected_mean[None]], axis=0)
            predicted = scorer.stacked_features()
            l2_scores = scorer.stacked_l2_scores()
            if len(predicted) != len(normalized):
                raise RuntimeError("Captured features do not match traced candidates plus selected mean")
            raw = denormalize_sequences(bundle, normalized)
            sim = simulate_sequences(state, raw, seed=episode_seed)
            enc_started = time.perf_counter()
            true = encode_endpoint_batch(bundle, sim.final_images, sim.final_states)
            encode_seconds = time.perf_counter() - enc_started
            np.savez_compressed(
                path,
                state=state,
                normalized_actions=normalized.astype(np.float32),
                predicted_features=predicted.astype(np.float32),
                true_features=true.astype(np.float32),
                l2_scores=l2_scores.astype(np.float32),
                simulator_cost=sim.costs,
                final_states=sim.final_states,
                contact_any=sim.contact_any,
                trace_scores=trace.scores,
                means_before=trace.means_before,
                stds_before=trace.stds_before,
                means_after=trace.means_after,
                stds_after=trace.stds_after,
                selected_index=np.int64(len(normalized) - 1),
            )
            timing = {
                "episode": episode, "seed": episode_seed,
                "cem_seconds": trace.runtime_seconds,
                "sim_seconds": sim.runtime_seconds,
                "encode_seconds": encode_seconds,
                "candidates": len(normalized),
            }
            (episode_dir / f"episode_{episode:03d}.timing.json").write_text(
                json.dumps(timing, indent=2), encoding="utf-8"
            )
        with np.load(path) as data:
            all_true.append(data["true_features"])
            all_pred.append(data["predicted_features"])
            all_cost.append(data["simulator_cost"])
            all_l2.append(data["l2_scores"])
            all_group.append(np.full(len(data["simulator_cost"]), episode, dtype=np.int32))
        timing_path = episode_dir / f"episode_{episode:03d}.timing.json"
        timings.append(json.loads(timing_path.read_text(encoding="utf-8")))
        print(f"development episode {episode + 1}/{cfg['episodes']} ready", flush=True)

    x_true = np.concatenate(all_true)
    x_pred = np.concatenate(all_pred)
    y = np.concatenate(all_cost)
    groups = np.concatenate(all_group)
    alphas = [float(x) for x in cfg["readout"]["alphas"]]
    folds = GroupKFold(n_splits=int(cfg["episodes"]))
    alpha_scores = {}
    for alpha in alphas:
        fold_mse = []
        for train, valid in folds.split(x_true, y, groups):
            model = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
            model.fit(x_true[train], y[train])
            fold_mse.append(float(np.mean((model.predict(x_true[valid]) - y[valid]) ** 2)))
        alpha_scores[str(alpha)] = {"fold_mse": fold_mse, "mean_mse": float(np.mean(fold_mse))}
    selected_alpha = min(alphas, key=lambda a: alpha_scores[str(a)]["mean_mse"])
    frozen = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=selected_alpha))])
    frozen.fit(x_true, y)
    scorer_path = Path(cfg["outputs"]["scorer"])
    scorer_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(frozen, scorer_path)
    true_score = frozen.predict(x_true)
    pred_score = frozen.predict(x_pred)
    episode_metrics = []
    for episode in range(int(cfg["episodes"])):
        mask = groups == episode
        cost = y[mask]
        denom = max(float(cost.max() - cost.min()), 1e-8)
        episode_metrics.append({
            "episode": episode,
            "cost_range": float(np.ptp(cost)),
            "gt_rho": float(spearmanr(true_score[mask], cost).statistic),
            "pred_rho": float(spearmanr(pred_score[mask], cost).statistic),
            "gt_regret": float((cost[np.argmin(true_score[mask])] - cost.min()) / denom),
            "pred_regret": float((cost[np.argmin(pred_score[mask])] - cost.min()) / denom),
            "l2_regret": float((cost[np.argmin(all_l2[episode])] - cost.min()) / denom),
        })
    digest = hashlib.sha256(scorer_path.read_bytes()).hexdigest().upper()
    torch_readout = TorchReadout.from_joblib(scorer_path, bundle.model.device)
    targeted_metrics = []
    for episode in range(int(cfg["targeted_validation_episodes"])):
        target_path = episode_dir / f"episode_{episode:03d}.targeted.npz"
        episode_seed = int(cfg["seed"]) + episode
        with np.load(episode_dir / f"episode_{episode:03d}.npz") as baseline:
            state = baseline["state"]
            baseline_selected_cost = float(baseline["simulator_cost"][-1])
        if not target_path.exists():
            z0 = encode_single_state(bundle, state)
            target_scorer = FeatureCapturingScorer(
                bundle, z_goal, readout=torch_readout,
                alpha=float(cfg["planner"]["terminal_proprio_alpha"]),
                model_query_batch_size=int(cfg["model"]["model_query_batch_size"]),
            )
            target_trace = _planner(cfg, bundle, episode_seed).plan(z0, target_scorer)
            target_actions = np.concatenate(
                [target_trace.flat_candidates, target_trace.selected_mean[None]], axis=0
            )
            target_sim = simulate_sequences(
                state, denormalize_sequences(bundle, target_actions), seed=episode_seed,
                collect_final_images=False,
            )
            np.savez_compressed(
                target_path,
                normalized_actions=target_actions.astype(np.float32),
                aligned_scores=target_scorer.stacked_returned_scores().astype(np.float32),
                simulator_cost=target_sim.costs,
                means_before=target_trace.means_before, stds_before=target_trace.stds_before,
                means_after=target_trace.means_after, stds_after=target_trace.stds_after,
            )
        with np.load(target_path) as target:
            target_selected_cost = float(target["simulator_cost"][-1])
            target_oracle_cost = float(target["simulator_cost"].min())
        targeted_metrics.append({
            "episode": episode,
            "baseline_selected_raw_cost": baseline_selected_cost,
            "targeted_selected_raw_cost": target_selected_cost,
            "targeted_oracle_raw_cost": target_oracle_cost,
            "raw_cost_improvement": baseline_selected_cost - target_selected_cost,
        })
    summary = {
        "development_only": True,
        "episodes": int(cfg["episodes"]),
        "candidates_per_episode": int(len(all_cost[0])),
        "selected_alpha": selected_alpha,
        "alpha_cv": alpha_scores,
        "scorer_sha256": digest,
        "episode_metrics": episode_metrics,
        "mean_gt_rho": float(np.nanmean([x["gt_rho"] for x in episode_metrics])),
        "mean_pred_rho": float(np.nanmean([x["pred_rho"] for x in episode_metrics])),
        "mean_gt_regret": float(np.mean([x["gt_regret"] for x in episode_metrics])),
        "mean_pred_regret": float(np.mean([x["pred_regret"] for x in episode_metrics])),
        "mean_l2_regret": float(np.mean([x["l2_regret"] for x in episode_metrics])),
        "targeted_integration_metrics": targeted_metrics,
        "timings": timings,
        "config": cfg,
    }
    (out / "development_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
