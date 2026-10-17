"""Configs reject impossible settings early and survive a checkpoint round trip."""

import pytest
import torch

from tiny_vit.config import (
    DATASET_PRESETS,
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_from_dict,
    config_to_dict,
)


def test_model_config_derived_sizes() -> None:
    config = ModelConfig(num_classes=10, image_size=128, patch_size=16, d_model=256, n_heads=8)
    assert config.num_patches == 64  # (128 / 16)^2
    assert config.head_dim == 32


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"image_size": 30, "patch_size": 4}, "multiple of patch_size"),
        ({"d_model": 30, "n_heads": 4}, "divisible by n_heads"),
        ({"patch_embed": "magic"}, "patch_embed"),
        ({"pool": "max"}, "pool"),
        ({"num_classes": 0}, "num_classes"),
    ],
)
def test_model_config_rejects_bad_values(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelConfig(**{"num_classes": 3, **kwargs})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dataset": "imagenet"},
        {"augment": "extreme"},
        {"val_fraction": 0.6, "test_fraction": 0.4},
        {"val_fraction": -0.1},
    ],
)
def test_data_config_rejects_bad_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"epochs": 0}, {"precision": "fp8"}, {"label_smoothing": 1.0}, {"batch_size": 0}]
)
def test_train_config_rejects_bad_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


def test_round_trip_through_a_safe_checkpoint(tmp_path) -> None:
    data = DataConfig(classes=("b", "a"), max_per_class=5)
    model = ModelConfig(num_classes=4, image_size=32, patch_size=8)
    path = tmp_path / "configs.pt"
    torch.save({"data": config_to_dict(data), "model": config_to_dict(model)}, path)
    loaded = torch.load(path, weights_only=True)  # only plain types: safe loading works
    assert config_from_dict(DataConfig, loaded["data"]) == data
    assert config_from_dict(ModelConfig, loaded["model"]) == model


def test_unknown_keys_are_ignored() -> None:
    values = config_to_dict(ModelConfig(num_classes=2)) | {"added_in_a_later_version": 1}
    assert config_from_dict(ModelConfig, values) == ModelConfig(num_classes=2)


@pytest.mark.parametrize("dataset", sorted(DATASET_PRESETS))
def test_every_preset_builds_a_valid_model_config(dataset: str) -> None:
    ModelConfig(num_classes=10, **DATASET_PRESETS[dataset])
