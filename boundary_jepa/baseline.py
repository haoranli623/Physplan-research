"""Load the official Push-T JEPA-WM without requiring the offline dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from app.plan_common.datasets import get_data_stats
from app.plan_common.datasets.preprocessor import Preprocessor
from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module


OFFICIAL_PUSHT_CONFIG = Path(
    "configs/evals/simu_env_planning/pt/jepa-wm/"
    "pt_L2_cem_sourcedset_H6_nas6_ctxt2_r224_alpha0.1_ep96_decode.yaml"
)


@dataclass(frozen=True)
class BaselineBundle:
    model: torch.nn.Module
    preprocessor: Preprocessor
    config: dict[str, Any]
    checkpoint: Path


def _build_preprocessor(config: dict[str, Any]) -> Preprocessor:
    model_kwargs = config["model_kwargs"]
    data_cfg = model_kwargs["data"]
    aug_cfg = model_kwargs["data_aug"]
    stats = get_data_stats("pusht")
    img_size = int(data_cfg.get("img_size", 224))
    normalize = aug_cfg.get(
        "normalize",
        [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
    )
    transform = make_transforms(
        img_size=img_size,
        normalize=normalize,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(img_size=img_size, normalize=normalize)
    return Preprocessor(
        action_mean=torch.tensor(stats["action_mean"], dtype=torch.float32),
        action_std=torch.tensor(stats["action_std"], dtype=torch.float32),
        state_mean=torch.tensor(stats["state_mean"], dtype=torch.float32),
        state_std=torch.tensor(stats["state_std"], dtype=torch.float32),
        proprio_mean=torch.tensor(stats["proprio_mean"], dtype=torch.float32),
        proprio_std=torch.tensor(stats["proprio_std"], dtype=torch.float32),
        transform=transform,
        inverse_transform=inverse_transform,
    )


def load_official_pusht_jepa(
    checkpoint: str | Path,
    device: str | torch.device = "cuda:0",
    config_path: str | Path = OFFICIAL_PUSHT_CONFIG,
) -> BaselineBundle:
    """Load the official pretrained model and its exact Push-T preprocessing.

    Decoder heads are intentionally disabled: the pilot evaluates frozen encoder
    latents and predictor latents, so downloading a large image decoder would add
    no scientific information.
    """

    checkpoint = Path(checkpoint).resolve()
    config_path = Path(config_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_kwargs = dict(config["model_kwargs"]["pretrain_kwargs"])
    model_kwargs["heads_cfg"] = {"architectures": {}, "pretrain_dec_path": None}
    data_cfg = config["model_kwargs"]["data"]
    wrapper_cfg = config["model_kwargs"].get("wrapper_kwargs", {})
    preprocessor = _build_preprocessor(config)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = init_module(
        folder=checkpoint.parent,
        checkpoint=checkpoint.name,
        model_kwargs=model_kwargs,
        device=torch_device,
        action_dim=2,
        proprio_dim=4,
        preprocessor=preprocessor,
        cfgs_data=data_cfg,
        wrapper_kwargs=wrapper_cfg,
    )
    model.eval()
    model.requires_grad_(False)
    return BaselineBundle(model=model, preprocessor=preprocessor, config=config, checkpoint=checkpoint)


def flatten_normalized_action_chunks(
    raw_actions: torch.Tensor,
    preprocessor: Preprocessor,
    frameskip: int = 5,
) -> torch.Tensor:
    """Convert raw simulator actions [B,T,2] to model chunks [H,B,2*frameskip]."""

    if raw_actions.ndim != 3 or raw_actions.shape[-1] != 2:
        raise ValueError(f"Expected [B,T,2] actions, got {tuple(raw_actions.shape)}")
    if raw_actions.shape[1] % frameskip:
        raise ValueError("Control-step count must be divisible by frameskip")
    normalized = preprocessor.normalize_actions(raw_actions)
    batch, steps, action_dim = normalized.shape
    return normalized.reshape(batch, steps // frameskip, frameskip * action_dim).transpose(0, 1).contiguous()
