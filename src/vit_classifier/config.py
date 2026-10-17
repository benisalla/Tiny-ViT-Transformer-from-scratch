"""Every hyperparameter of the project, in three small dataclasses.

DataConfig   what images we read and how we prepare them
ModelConfig  the shape of the Vision Transformer
TrainConfig  how we optimise it

The CLIs (`vit-train`, ...) build these from flags; checkpoints store them as
plain dicts so a saved model can always be rebuilt exactly.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

# ImageNet statistics: the usual choice, and close enough for most photos.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DATASETS = ("shapes", "cifar10", "folder")
AUGMENTATIONS = ("none", "light", "strong")
PATCH_EMBEDDINGS = ("linear", "conv")
POOLINGS = ("cls", "mean")

# The 12 tomato / potato classes of PlantVillage the first version of this
# project was trained on (folder names of the PlantVillage "color" release).
PLANTVILLAGE_TOMATO_POTATO = (
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)


@dataclass
class DataConfig:
    """Where the images come from and how they are turned into tensors."""

    # "shapes": synthetic, generated on the fly (no download)
    # "cifar10": the classic 10-class 32x32 benchmark (downloaded by torchvision)
    # "folder": any folder laid out as data_dir/<class name>/<image files>
    dataset: str = "folder"
    data_dir: str = "data/plantvillage"
    # Only these class folders (None = every sub-folder of data_dir).
    classes: tuple[str, ...] | None = None
    # Cap on images per class (None = all). Handy for quick experiments.
    max_per_class: int | None = None
    # Fractions of each class held out (folder / shapes). CIFAR-10 has its own test set.
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 0  # the split depends only on this, never on the training seed
    image_size: int = 128
    augment: str = "light"
    image_mean: tuple[float, float, float] = IMAGENET_MEAN
    image_std: tuple[float, float, float] = IMAGENET_STD
    shapes_per_class: int = 2000  # synthetic images per class for "shapes"

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"dataset must be one of {DATASETS}, got {self.dataset!r}")
        if self.augment not in AUGMENTATIONS:
            raise ValueError(f"augment must be one of {AUGMENTATIONS}, got {self.augment!r}")
        if not (0 <= self.val_fraction < 1 and 0 <= self.test_fraction < 1):
            raise ValueError("val_fraction and test_fraction must be in [0, 1)")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must leave some training data")
        if self.classes is not None:
            self.classes = tuple(self.classes)  # lists come back from checkpoints
        self.image_mean = tuple(self.image_mean)
        self.image_std = tuple(self.image_std)


@dataclass
class ModelConfig:
    """Shape of the Vision Transformer.

    The image is cut into (image_size / patch_size)^2 square patches; each patch
    becomes one token of width d_model.
    """

    num_classes: int
    image_size: int = 128
    patch_size: int = 16
    in_channels: int = 3
    d_model: int = 256  # token width
    n_layers: int = 6  # Transformer blocks
    n_heads: int = 8  # attention heads (head size = d_model / n_heads)
    mlp_ratio: float = 4.0  # hidden width of the MLP = mlp_ratio * d_model
    dropout: float = 0.1
    # "linear": flatten each patch, LayerNorm, Linear (the first version's default)
    # "conv":   one Conv2d with kernel = stride = patch_size (same maths, one op)
    patch_embed: str = "linear"
    # "cls": classify from the extra [CLS] token (original ViT)
    # "mean": classify from the average of the patch tokens
    pool: str = "cls"

    def __post_init__(self) -> None:
        if self.num_classes < 1:
            raise ValueError("num_classes must be >= 1")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be a multiple of patch_size "
                f"({self.patch_size}), otherwise border pixels are silently dropped"
            )
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )
        if self.patch_embed not in PATCH_EMBEDDINGS:
            raise ValueError(f"patch_embed must be one of {PATCH_EMBEDDINGS}")
        if self.pool not in POOLINGS:
            raise ValueError(f"pool must be one of {POOLINGS}")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainConfig:
    """How the model is optimised."""

    output_dir: str = "checkpoints/run"
    epochs: int = 30
    batch_size: int = 64  # per process (per GPU)
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps * GPUs
    lr: float = 5e-4  # peak learning rate, reached after warmup
    weight_decay: float = 0.05
    warmup_steps: int = 500
    label_smoothing: float = 0.1
    grad_clip: float = 1.0  # max gradient norm (0 = off)
    precision: str = "auto"  # auto | fp32 | bf16 | fp16
    device: str = "auto"  # auto | cpu | cuda | mps
    num_workers: int = 4
    seed: int = 0
    log_every: int = 50  # optimizer steps between log lines
    eval_every: int = 0  # optimizer steps between validations (0 = once per epoch)
    resume: str | None = None  # checkpoint to continue from
    overfit: int | None = None  # train AND validate on the first N training images

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("epochs, batch_size and grad_accum_steps must be >= 1")
        if self.precision not in ("auto", "fp32", "bf16", "fp16"):
            raise ValueError(f"unknown precision {self.precision!r}")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")


# Sensible defaults per dataset, used by the CLI when a flag is not given.
DATASET_PRESETS: dict[str, dict[str, Any]] = {
    # 32x32 images, 4x4 patches -> 64 tokens. Small enough for a laptop CPU.
    "shapes": {"image_size": 32, "patch_size": 4, "d_model": 128, "n_layers": 4, "n_heads": 4},
    "cifar10": {"image_size": 32, "patch_size": 4, "d_model": 192, "n_layers": 6, "n_heads": 6},
    # Leaf photos are 256x256; 128 px with 16 px patches -> 64 tokens.
    "folder": {"image_size": 128, "patch_size": 16, "d_model": 256, "n_layers": 6, "n_heads": 8},
}


def config_to_dict(config: Any) -> dict[str, Any]:
    """Dataclass -> plain dict (only built-in types: safe for torch.load(weights_only=True))."""
    return dataclasses.asdict(config)


def config_from_dict(cls: type, values: dict[str, Any]) -> Any:
    """Inverse of config_to_dict. Unknown keys are ignored (forward compatibility)."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in values.items() if k in names})


__all__ = [
    "DATASETS",
    "DATASET_PRESETS",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "PLANTVILLAGE_TOMATO_POTATO",
    "DataConfig",
    "ModelConfig",
    "TrainConfig",
    "config_from_dict",
    "config_to_dict",
]
