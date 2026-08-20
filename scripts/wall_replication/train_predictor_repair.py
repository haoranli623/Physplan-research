"""Develop a minimal standard-objective predictor repair on D_wall_dev only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from tensordict import TensorDict
import yaml

from wall_replication.baseline import flatten_normalized_action_chunks, load_official_wall_jepa
from wall_replication.environment import render_states_224
from wall_replication.features import _obs_tensordict


def render_mixed(states: np.ndarray, wall_x: np.ndarray, door_y: np.ndarray) -> np.ndarray:
    output = np.empty((*states.shape[:-1], 224, 224, 3), dtype=np.uint8)
    layouts = np.stack([wall_x, door_y], axis=1)
    for layout in np.unique(layouts, axis=0):
        mask = np.all(layouts == layout, axis=1)
        output[mask] = render_states_224(
            states[mask], wall_x=int(layout[0]), door_y=int(layout[1])
        )
    return output


def batches(indices: np.ndarray, size: int):
    for start in range(0, len(indices), size):
        yield indices[start : start + size]


def prepare_batch(data, flat_indices: np.ndarray, candidate_count: int):
    anchors = flat_indices // candidate_count
    candidates = flat_indices % candidate_count
    states = data["sampled_states"][anchors, candidates]
    wall_x = data["wall_x"][anchors]
    door_y = data["door_y"][anchors]
    initial_images = render_mixed(states[:, :1], wall_x, door_y)
    future_images = render_mixed(states[:, 1:], wall_x, door_y)
    return (
        anchors,
        candidates,
        states,
        initial_images,
        future_images,
        data["actions"][anchors, candidates],
    )


def predict_batch(bundle, data, flat_indices: np.ndarray, candidate_count: int, grad: bool):
    anchors, candidates, states, initial_images, future_images, actions = prepare_batch(
        data, flat_indices, candidate_count
    )
    device = bundle.model.device
    chunks = flatten_normalized_action_chunks(
        torch.from_numpy(actions), bundle.preprocessor, frameskip=5
    ).to(device)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    with torch.inference_mode(), amp:
        z0 = bundle.model.encode(_obs_tensordict(initial_images, states[:, :1]).to(device))
        zt = bundle.model.encode(_obs_tensordict(future_images, states[:, 1:]).to(device))
    context = torch.enable_grad() if grad else torch.inference_mode()
    with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        zp = bundle.model.unroll(z0, chunks)
        horizon = states.shape[1] - 1
        pred_visual = zp["visual"][-horizon:].transpose(0, 1)
        pred_proprio = zp["proprio"][-horizon:].transpose(0, 1)
        visual_loss = (pred_visual - zt["visual"]).float().pow(2).mean()
        proprio_loss = (pred_proprio - zt["proprio"]).float().pow(2).mean()
        loss = visual_loss + 0.1 * proprio_loss
    return loss, visual_loss, proprio_loss


def evaluate(bundle, data, anchor_indices: np.ndarray, batch_size: int) -> dict[str, float]:
    candidate_count = data["simulator_cost"].shape[1]
    flat = np.concatenate(
        [np.arange(anchor * candidate_count, (anchor + 1) * candidate_count) for anchor in anchor_indices]
    )
    weighted = np.zeros(3, dtype=np.float64)
    count = 0
    bundle.model.model.predictor.eval()
    for batch in batches(flat, batch_size):
        loss, visual, proprio = predict_batch(bundle, data, batch, candidate_count, grad=False)
        weighted += np.asarray([loss.item(), visual.item(), proprio.item()]) * len(batch)
        count += len(batch)
    return {
        "loss": float(weighted[0] / count),
        "visual_loss": float(weighted[1] / count),
        "proprio_loss": float(weighted[2] / count),
    }


def train_epochs(bundle, data, anchor_indices: np.ndarray, cfg: dict, epochs: int, seed: int):
    predictor = bundle.model.model.predictor
    predictor.requires_grad_(True)
    predictor.train()
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    candidate_count = data["simulator_cost"].shape[1]
    flat = np.concatenate(
        [np.arange(anchor * candidate_count, (anchor + 1) * candidate_count) for anchor in anchor_indices]
    )
    rng = np.random.default_rng(seed)
    logs = []
    for epoch in range(1, epochs + 1):
        rng.shuffle(flat)
        total = 0.0
        seen = 0
        for step, batch in enumerate(batches(flat, int(cfg["batch_size"])), start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = predict_batch(bundle, data, batch, candidate_count, grad=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
            seen += len(batch)
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} train_loss={total / seen:.6f}", flush=True)
        logs.append({"epoch": epoch, "train_loss": total / seen})
    predictor.eval()
    predictor.requires_grad_(False)
    return logs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wall_replication/development.yaml")
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    repair = config["predictor_repair_development"]
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = np.load(config["outputs"]["cache"])
    train_count = int(repair["train_anchor_count"])
    val_count = int(repair["validation_anchor_count"])
    train_anchors = np.arange(train_count)
    val_anchors = np.arange(train_count, train_count + val_count)

    bundle = load_official_wall_jepa(config["checkpoint"], device=config["device"])
    history = [{"epoch": 0, "validation": evaluate(bundle, data, val_anchors, int(repair["batch_size"]))}]
    predictor = bundle.model.model.predictor
    predictor.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(repair["learning_rate"]),
        weight_decay=float(repair["weight_decay"]),
    )
    candidate_count = data["simulator_cost"].shape[1]
    flat_train = np.arange(train_count * candidate_count)
    rng = np.random.default_rng(seed)
    best_epoch = 0
    best_loss = history[0]["validation"]["loss"]
    best_state = {key: value.detach().cpu().clone() for key, value in predictor.state_dict().items()}
    for epoch in range(1, int(repair["maximum_epochs"]) + 1):
        predictor.train()
        predictor.requires_grad_(True)
        rng.shuffle(flat_train)
        total = 0.0
        seen = 0
        for step, batch in enumerate(batches(flat_train, int(repair["batch_size"])), start=1):
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = predict_batch(bundle, data, batch, candidate_count, grad=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch)
            seen += len(batch)
            if step % 50 == 0:
                print(f"epoch={epoch} step={step} train_loss={total / seen:.6f}", flush=True)
        predictor.eval()
        predictor.requires_grad_(False)
        validation = evaluate(bundle, data, val_anchors, int(repair["batch_size"]))
        entry = {"epoch": epoch, "train_loss": total / seen, "validation": validation}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in predictor.state_dict().items()}

    out = Path("results/wall_replication/development")
    # Store the held-out-development selected state. A protocol refit is unnecessary:
    # these 48 training anchors are already disjoint from diagnosis and repair.
    checkpoint = out / "predictor_repair_dev_selected.pth.tar"
    torch.save(
        {
            "predictor": best_state,
            "selected_epoch": best_epoch,
            "validation_loss": best_loss,
            "train_anchors": train_anchors.tolist(),
            "validation_anchors": val_anchors.tolist(),
            "source_checkpoint": str(Path(config["checkpoint"]).resolve()),
        },
        checkpoint,
    )
    result = {
        "development_only": True,
        "selection_metric": repair["checkpoint_selection"],
        "selected_epoch": best_epoch,
        "selected_validation_loss": best_loss,
        "history": history,
        "checkpoint": str(checkpoint),
    }
    (out / "predictor_repair_development.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
