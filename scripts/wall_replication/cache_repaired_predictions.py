"""Apply the frozen development predictor repair to an existing Wall split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from wall_replication.baseline import flatten_normalized_action_chunks, load_official_wall_jepa
from wall_replication.environment import render_states_224
from wall_replication.features import _obs_tensordict, _pool_proprio, _pool_visual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wall_replication/development.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--repair-checkpoint",
        default="results/wall_replication/development/predictor_repair_dev_selected.pth.tar",
    )
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    source_path = Path(args.source or cfg["outputs"]["cache"])
    source = np.load(source_path)
    bundle = load_official_wall_jepa(cfg["checkpoint"], device=cfg["device"])
    repair = torch.load(args.repair_checkpoint, map_location="cpu", weights_only=False)
    message = bundle.model.model.predictor.load_state_dict(repair["predictor"], strict=True)
    bundle.model.eval()
    anchors, candidates = source["simulator_cost"].shape
    horizon = int(cfg["environment"]["latent_horizon"])
    pred_visual = np.empty((anchors, candidates, horizon, 384), dtype=np.float32)
    pred_proprio = np.empty((anchors, candidates, horizon, 16), dtype=np.float32)
    default_score = np.empty((anchors, candidates), dtype=np.float32)
    device = bundle.model.device
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for anchor in range(anchors):
            wall_x = int(source["wall_x"][anchor])
            door_y = int(source["door_y"][anchor])
            start = source["sampled_states"][anchor, 0, 0]
            goal = source["goals"][anchor]
            initial_image = render_states_224(start[None], wall_x=wall_x, door_y=door_y)[None]
            goal_image = render_states_224(goal[None], wall_x=wall_x, door_y=door_y)[None]
            z0 = bundle.model.encode(
                _obs_tensordict(initial_image, start[None, None]).to(device)
            )
            zg = bundle.model.encode(_obs_tensordict(goal_image, goal[None, None]).to(device))
            chunks = flatten_normalized_action_chunks(
                torch.from_numpy(source["actions"][anchor]), bundle.preprocessor, frameskip=5
            ).to(device)
            zp = bundle.model.unroll(z0, chunks)
            visual = zp["visual"][-horizon:].transpose(0, 1)
            proprio = zp["proprio"][-horizon:].transpose(0, 1)
            goal_v = zg["visual"][:, -1]
            goal_p = zg["proprio"][:, -1]
            score = (visual[:, -1] - goal_v).float().pow(2).mean(dim=(1, 2, 3, 4))
            score += 0.1 * (proprio[:, -1] - goal_p).float().pow(2).mean(
                dim=tuple(range(1, proprio.ndim - 1))
            )
            pred_visual[anchor] = _pool_visual(visual).float().cpu().numpy()
            pred_proprio[anchor] = _pool_proprio(proprio).float().cpu().numpy()
            default_score[anchor] = score.cpu().numpy()
            print(f"Repaired predictions {anchor + 1}/{anchors}", flush=True)
    elapsed = time.perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source=str(source_path.resolve()),
        pred_visual=pred_visual,
        pred_proprio=pred_proprio,
        default_pred_score=default_score,
    )
    metadata = {
        "source": str(source_path.resolve()),
        "repair_checkpoint": str(Path(args.repair_checkpoint).resolve()),
        "load_state_dict": str(message),
        "selected_epoch": int(repair["selected_epoch"]),
        "anchors": anchors,
        "candidates": anchors * candidates,
        "seconds": elapsed,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
