"""Episode-level analysis for the frozen full-sequence experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def bootstrap(values: np.ndarray, samples: int, seed: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
        "episodes": int(len(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_sequence/protocol.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    directory = Path(cfg["outputs"]["episode_directory"])
    rows = []
    for episode in range(int(cfg["final_episodes"])):
        path = directory / f"episode_{episode:03d}.json"
        raw_path = directory / f"episode_{episode:03d}.npz"
        if not path.exists() or not raw_path.exists():
            raise RuntimeError(
                f"Missing paired final episode artifacts: json={path.exists()} npz={raw_path.exists()} "
                f"for episode {episode}"
            )
        item = json.loads(path.read_text(encoding="utf-8"))
        expected_seed = int(cfg["final_seed"]) + episode
        if int(item.get("episode", -1)) != episode or int(item.get("seed", -1)) != expected_seed:
            raise RuntimeError(
                f"Episode identity mismatch for {path}: expected episode={episode}, seed={expected_seed}; "
                f"got episode={item.get('episode')}, seed={item.get('seed')}"
            )
        fixed, search, e2e = item["fixed_trace"], item["search"], item["end_to_end"]
        rows.append({
            "episode": episode, "seed": item["seed"],
            "gt_readout_regret": fixed["regrets"]["gt_readout"],
            "pred_readout_regret": fixed["regrets"]["pred_readout"],
            "pred_l2_regret": fixed["regrets"]["pred_l2"],
            "prediction_gap": fixed["prediction_gap"], "metric_gap": fixed["metric_gap"],
            "baseline_regret": e2e["baseline_regret"], "targeted_regret": e2e["targeted_regret"],
            "repair_improvement": e2e["repair_improvement"],
            "baseline_raw_cost": e2e["baseline_raw_cost"], "targeted_raw_cost": e2e["targeted_raw_cost"],
            "baseline_selection_gap_raw": search["baseline_selection_gap_raw"],
            "targeted_selection_gap_raw": search["targeted_selection_gap_raw"],
            "baseline_coverage_gap_raw": search["baseline_coverage_gap_to_union_raw"],
            "targeted_coverage_gap_raw": search["targeted_coverage_gap_to_union_raw"],
        })
    frame = pd.DataFrame(rows)
    result_dir = Path(cfg["outputs"]["result_directory"])
    result_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(result_dir / "episode_metrics.csv", index=False)
    samples = int(cfg["statistics"]["bootstrap_samples"])
    seed = int(cfg["statistics"]["bootstrap_seed"])
    keys = [x for x in frame.columns if x not in {"episode", "seed"}]
    summaries = {key: bootstrap(frame[key].to_numpy(), samples, seed + i) for i, key in enumerate(keys)}
    tolerance = float(cfg["statistics"]["unchanged_tolerance"])
    improvements = frame["repair_improvement"].to_numpy()
    counts = {
        "helped": int(np.sum(improvements > tolerance)),
        "unchanged": int(np.sum(np.abs(improvements) <= tolerance)),
        "harmed": int(np.sum(improvements < -tolerance)),
    }
    pred, metric = summaries["prediction_gap"], summaries["metric_gap"]
    repair = summaries["repair_improvement"]
    margin = float(cfg["interpretation"]["material_gap"])
    ratio = float(cfg["interpretation"]["dominance_ratio"])
    metric_dominant = metric["ci_low"] > margin and metric["mean"] > ratio * max(pred["mean"], margin)
    pred_dominant = pred["ci_low"] > margin and pred["mean"] > ratio * max(metric["mean"], margin)
    if metric_dominant and repair["ci_low"] > 0:
        decision = "STRONG VALIDATION"
    elif metric_dominant:
        decision = "METRIC HEADROOM WITHOUT END-TO-END REPAIR"
    elif pred_dominant:
        decision = "PREDICTION BECOMES IMPORTANT"
    elif summaries["baseline_coverage_gap_raw"]["ci_low"] > margin:
        decision = "SEARCH BECOMES DOMINANT"
    else:
        decision = "NO CLEAR SIGNAL"
    summary = {
        "decision": decision,
        "episodes": len(frame),
        "summaries": summaries,
        "repair_counts": counts,
        "protocol": cfg,
        "warning": "Headroom terms are paired contrasts and are not additive causal components.",
    }
    serialized = json.dumps(summary, indent=2, default=str)
    (result_dir / "summary.json").write_text(serialized, encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
