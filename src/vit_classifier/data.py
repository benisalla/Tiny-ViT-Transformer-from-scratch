"""Datasets and image transforms.

Three sources, one interface (`build_datasets` -> train / val / test + class names):

- "shapes":  synthetic images of 6 shapes in random colours, sizes and positions,
             generated on the fly (no download). The class depends only on the
             shape, so a model that scores well must have read the pixels.
- "cifar10": the classic 10-class 32x32 benchmark (downloaded by torchvision).
- "folder":  any directory laid out as data_dir/<class name>/<image files>, e.g.
             PlantVillage (see docs/datasets.md for the download command).

Splits are stratified (same class balance in train / val / test) and depend
only on DataConfig.split_seed, so evaluation always sees the same test images.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from vit_classifier.config import DataConfig

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# 1. Transforms
# ---------------------------------------------------------------------------


def build_transform(config: DataConfig, train: bool) -> Callable[[Image.Image], torch.Tensor]:
    """PIL image -> normalised (3, image_size, image_size) tensor.

    Evaluation always uses the plain resize. Training adds augmentation:
    "light"  random crop (70-100% of the image) + horizontal flip
    "strong" also vertical flip, TrivialAugment and random erasing: ViTs have
             no built-in image prior, so from scratch they need a lot of it.
    """
    size = config.image_size
    normalize = [transforms.ToTensor(), transforms.Normalize(config.image_mean, config.image_std)]
    if not train or config.augment == "none":
        return transforms.Compose([transforms.Resize((size, size)), *normalize])
    if config.augment == "light":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            *normalize,
        ])  # fmt: skip
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0), antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.TrivialAugmentWide(),
        *normalize,
        transforms.RandomErasing(p=0.25),
    ])  # fmt: skip


def denormalize(pixels: torch.Tensor, config: DataConfig) -> torch.Tensor:
    """Undo Normalize: (3, H, W) model input -> values in [0, 1] for display."""
    mean = torch.tensor(config.image_mean).view(3, 1, 1)
    std = torch.tensor(config.image_std).view(3, 1, 1)
    return (pixels.cpu() * std + mean).clamp(0, 1)


# ---------------------------------------------------------------------------
# 2. One dataset class for every source
# ---------------------------------------------------------------------------


class ImageDataset(Dataset):
    """A list of (key, label) pairs; `load(key)` returns the PIL image.

    The key is a file path (folder), an index (shapes, CIFAR-10): the loading
    differs, the rest is shared.
    """

    def __init__(
        self,
        items: Sequence[tuple[object, int]],
        load: Callable[[object], Image.Image],
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.items = list(items)
        self.load = load
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        key, label = self.items[index]
        return self.transform(self.load(key).convert("RGB")), label

    @property
    def labels(self) -> list[int]:
        return [label for _, label in self.items]


def open_image(path: object) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")  # convert() reads the pixels before the file closes


def stratified_split(
    items: Sequence[tuple[object, int]],
    fractions: Sequence[float],
    seed: int,
) -> list[list[tuple[object, int]]]:
    """Split items into len(fractions) + 1 parts with the same class balance.

    fractions are the held-out parts (e.g. val, test); the rest is the first
    part (train). Every class keeps at least one training item.
    """
    by_class: dict[int, list] = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)
    rng = random.Random(seed)
    parts: list[list] = [[] for _ in range(len(fractions) + 1)]
    for label in sorted(by_class):
        group = by_class[label][:]
        rng.shuffle(group)
        start = 0
        for i, fraction in enumerate(fractions, start=1):
            n = min(round(fraction * len(group)), len(group) - start - 1)
            parts[i].extend(group[start : start + max(n, 0)])
            start += max(n, 0)
        parts[0].extend(group[start:])
    return parts


def balanced_subset(items: Sequence[tuple[object, int]], n: int) -> list[tuple[object, int]]:
    """The first n items taken round-robin over the classes (class 0, 1, 2, 0, 1, ...)."""
    by_class: dict[int, list] = defaultdict(list)
    for item in items:
        by_class[item[1]].append(item)
    queues = [by_class[label] for label in sorted(by_class)]
    subset: list[tuple[object, int]] = []
    for i in range(max(map(len, queues), default=0)):
        subset.extend(queue[i] for queue in queues if i < len(queue))
    return subset[:n]


# ---------------------------------------------------------------------------
# 3. Sources
# ---------------------------------------------------------------------------

# --- synthetic shapes ---

SHAPE_CLASSES = ("circle", "square", "triangle", "diamond", "cross", "ring")
SHAPE_CANVAS = 64  # rendered at 64x64; the transform resizes to image_size
_COLORS = [
    (230, 60, 60), (60, 180, 75), (60, 110, 230), (255, 200, 40),
    (240, 130, 40), (160, 80, 200), (70, 210, 210), (240, 80, 180),
]  # fmt: skip


class ShapeRenderer:
    """Draws image `index` deterministically: same (seed, index) -> same picture."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(self, index: object) -> Image.Image:
        index = int(index)
        rng = random.Random(self.seed * 1_000_003 + index)
        kind = SHAPE_CLASSES[index % len(SHAPE_CLASSES)]
        shade = rng.randint(0, 60) if rng.random() < 0.5 else rng.randint(190, 255)
        background = tuple(min(255, max(0, shade + rng.randint(-15, 15))) for _ in range(3))
        image = Image.new("RGB", (SHAPE_CANVAS, SHAPE_CANVAS), background)
        draw = ImageDraw.Draw(image)
        r = rng.uniform(9, 22)  # half-size of the shape
        cx = rng.uniform(r + 2, SHAPE_CANVAS - r - 2)
        cy = rng.uniform(r + 2, SHAPE_CANVAS - r - 2)
        color = rng.choice(_COLORS)
        box = (cx - r, cy - r, cx + r, cy + r)
        if kind == "circle":
            draw.ellipse(box, fill=color)
        elif kind == "ring":
            draw.ellipse(box, outline=color, width=max(2, round(r / 3)))
        elif kind == "square":
            draw.rectangle(box, fill=color)
        elif kind == "triangle":
            draw.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)], fill=color)
        elif kind == "diamond":
            draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=color)
        else:  # cross
            w = r / 2.5
            draw.rectangle((cx - w, cy - r, cx + w, cy + r), fill=color)
            draw.rectangle((cx - r, cy - w, cx + r, cy + w), fill=color)
        return image


def shapes_items(config: DataConfig) -> list[tuple[object, int]]:
    total = config.shapes_per_class * len(SHAPE_CLASSES)
    return [(i, i % len(SHAPE_CLASSES)) for i in range(total)]


# --- image folder ---


def find_classes(data_dir: Path, classes: Sequence[str] | None) -> list[str]:
    """Class names = sub-folder names, sorted (or the requested subset, checked)."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"data directory not found: {data_dir} (expected data_dir/<class name>/<images>; "
            "see docs/datasets.md)"
        )
    available = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    if classes is None:
        if not available:
            raise FileNotFoundError(f"no class sub-folders in {data_dir}")
        return available
    missing = [c for c in classes if c not in available]
    if missing:
        raise FileNotFoundError(f"class folders not found in {data_dir}: {missing}")
    return sorted(classes)


def folder_items(
    data_dir: Path, class_names: Sequence[str], max_per_class: int | None, seed: int
) -> list[tuple[object, int]]:
    items = []
    rng = random.Random(seed)
    for label, name in enumerate(class_names):
        files = sorted(
            p for p in (data_dir / name).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise FileNotFoundError(f"no images in {data_dir / name}")
        if max_per_class is not None and len(files) > max_per_class:
            files = sorted(rng.sample(files, max_per_class))  # random subset, not the first N
        items.extend((str(p), label) for p in files)
    return items


# --- CIFAR-10 ---


class CifarImage:
    """Image `index` of a torchvision CIFAR-10 split (picklable, for DataLoader workers)."""

    def __init__(self, root: str, train: bool) -> None:
        from torchvision.datasets import CIFAR10

        self.dataset = CIFAR10(root, train=train, download=True)

    def __call__(self, index: object) -> Image.Image:
        return self.dataset[int(index)][0]


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------


@dataclass
class Datasets:
    train: ImageDataset
    val: ImageDataset
    test: ImageDataset
    class_names: list[str]


def build_datasets(config: DataConfig) -> Datasets:
    """Train / val / test datasets for the configured source."""
    train_tf, eval_tf = build_transform(config, train=True), build_transform(config, train=False)

    if config.dataset == "cifar10":
        train_source = CifarImage(config.data_dir, train=True)
        test_source = CifarImage(config.data_dir, train=False)
        class_names = list(train_source.dataset.classes)
        items = list(enumerate(train_source.dataset.targets))
        train, val = stratified_split(items, [config.val_fraction], config.split_seed)
        test = list(enumerate(test_source.dataset.targets))
        return Datasets(
            ImageDataset(train, train_source, train_tf),
            ImageDataset(val, train_source, eval_tf),
            ImageDataset(test, test_source, eval_tf),
            class_names,
        )

    if config.dataset == "shapes":
        class_names = list(SHAPE_CLASSES)
        items = shapes_items(config)
        load: Callable[[object], Image.Image] = ShapeRenderer(config.split_seed)
    else:
        data_dir = Path(config.data_dir)
        class_names = find_classes(data_dir, config.classes)
        items = folder_items(data_dir, class_names, config.max_per_class, config.split_seed)
        load = open_image

    fractions = [config.val_fraction, config.test_fraction]
    train, val, test = stratified_split(items, fractions, config.split_seed)
    return Datasets(
        ImageDataset(train, load, train_tf),
        ImageDataset(val, load, eval_tf),
        ImageDataset(test, load, eval_tf),
        class_names,
    )
