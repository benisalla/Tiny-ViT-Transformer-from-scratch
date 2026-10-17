"""Shared helpers: tiny configs, a fake image folder and a ready-made checkpoint.

Everything runs on CPU, offline, in seconds.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from tiny_vit.config import DataConfig, ModelConfig, TrainConfig
from tiny_vit.model import VisionTransformer
from tiny_vit.train import TrainState, build_optimizer, build_scheduler, save_checkpoint

COLORS = {"red": (220, 30, 30), "green": (30, 200, 30), "blue": (30, 30, 220)}


def tiny_model_config(**overrides) -> ModelConfig:
    values = dict(
        num_classes=3, image_size=16, patch_size=4, d_model=32, n_layers=2, n_heads=4,
        dropout=0.0,
    )  # fmt: skip
    values.update(overrides)
    return ModelConfig(**values)


def tiny_train_config(output_dir: Path, **overrides) -> TrainConfig:
    values = dict(
        output_dir=str(output_dir), epochs=2, batch_size=4, lr=3e-3, warmup_steps=2,
        num_workers=0, log_every=1, device="cpu", precision="fp32", label_smoothing=0.0,
        weight_decay=0.0,
    )  # fmt: skip
    values.update(overrides)
    return TrainConfig(**values)


def make_image_folder(root: Path, per_class: int = 10, size: int = 20) -> Path:
    """root/<red|green|blue>/<n>.png: solid colour images with a little noise.

    The class is the colour, so a model can learn it in a few steps.
    """
    generator = torch.Generator().manual_seed(0)
    for name, rgb in COLORS.items():
        (root / name).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            noise = torch.randint(-20, 21, (size, size, 3), generator=generator)
            pixels = (torch.tensor(rgb) + noise).clamp(0, 255).to(torch.uint8).numpy()
            Image.fromarray(pixels).save(root / name / f"{i:03d}.png")
    return root


@pytest.fixture
def image_folder(tmp_path: Path) -> Path:
    return make_image_folder(tmp_path / "images")


def folder_data_config(data_dir: Path, **overrides) -> DataConfig:
    values = dict(dataset="folder", data_dir=str(data_dir), image_size=16, augment="none")
    values.update(overrides)
    return DataConfig(**values)


def write_checkpoint(
    directory: Path, data_dir: Path | None = None, class_names: list[str] | None = None
) -> Path:
    """A valid vit-train checkpoint of a random tiny model (no training needed)."""
    class_names = class_names or sorted(COLORS)  # = sorted folder names of image_folder
    torch.manual_seed(0)
    model = VisionTransformer(tiny_model_config(num_classes=len(class_names)))
    config = tiny_train_config(directory)
    optimizer = build_optimizer(model, config, torch.device("cpu"))
    scheduler = build_scheduler(optimizer, 1, 10)
    data_config = folder_data_config(data_dir or directory / "missing")
    path = directory / "best.pt"
    save_checkpoint(
        path, model, optimizer, scheduler, torch.amp.GradScaler("cuda", enabled=False),
        TrainState(), data_config, config, class_names,
    )  # fmt: skip
    return path
