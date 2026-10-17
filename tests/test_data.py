"""Datasets, splits and transforms."""

from collections import Counter
from pathlib import Path

import pytest
import torch
from conftest import folder_data_config
from PIL import Image

import tiny_vit.data as data
from tiny_vit.config import DataConfig
from tiny_vit.data import (
    SHAPE_CLASSES,
    ShapeRenderer,
    balanced_subset,
    build_datasets,
    build_transform,
    denormalize,
    stratified_split,
)

# --- splits -----------------------------------------------------------------------


def test_stratified_split_is_disjoint_complete_balanced_and_seeded() -> None:
    items = [(i, i % 3) for i in range(300)]
    train, val, test = stratified_split(items, [0.1, 0.2], seed=0)
    assert sorted(train + val + test) == sorted(items)  # nothing lost, nothing duplicated
    assert Counter(label for _, label in val) == {0: 10, 1: 10, 2: 10}
    assert Counter(label for _, label in test) == {0: 20, 1: 20, 2: 20}
    assert stratified_split(items, [0.1, 0.2], seed=0) == [train, val, test]
    assert stratified_split(items, [0.1, 0.2], seed=1) != [train, val, test]


def test_tiny_classes_keep_a_training_example() -> None:
    train, val, test = stratified_split([("a", 0), ("b", 1), ("c", 1)], [0.5, 0.5], seed=0)
    assert {label for _, label in train} == {0, 1}


def test_balanced_subset_takes_every_class_in_turn() -> None:
    items = [("a0", 0), ("a1", 0), ("a2", 0), ("b0", 1), ("c0", 2), ("c1", 2)]
    assert balanced_subset(items, 5) == [("a0", 0), ("b0", 1), ("c0", 2), ("a1", 0), ("c1", 2)]
    assert balanced_subset(items, 100) == balanced_subset(items, 6)


# --- folder -----------------------------------------------------------------------


def test_folder_dataset(image_folder: Path) -> None:
    datasets = build_datasets(folder_data_config(image_folder))
    assert datasets.class_names == ["blue", "green", "red"]  # sorted folder names
    assert (len(datasets.train), len(datasets.val), len(datasets.test)) == (24, 3, 3)
    pixels, label = datasets.train[0]
    assert pixels.shape == (3, 16, 16) and pixels.dtype == torch.float32
    assert 0 <= label < 3
    # the label really is the folder: blue images have the largest blue channel
    for pixels, label in (datasets.test[i] for i in range(3)):
        channel = denormalize(pixels, DataConfig()).mean(dim=(1, 2)).argmax().item()
        assert datasets.class_names[label] == ["red", "green", "blue"][channel]


def test_folder_class_subset_and_cap(image_folder: Path) -> None:
    config = folder_data_config(image_folder, classes=("red", "blue"), max_per_class=5)
    datasets = build_datasets(config)
    assert datasets.class_names == ["blue", "red"]
    total = len(datasets.train) + len(datasets.val) + len(datasets.test)
    assert total == 10


def test_folder_errors_are_explained(tmp_path: Path, image_folder: Path) -> None:
    with pytest.raises(FileNotFoundError, match="data directory not found"):
        build_datasets(folder_data_config(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="purple"):
        build_datasets(folder_data_config(image_folder, classes=("purple",)))
    (image_folder / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="no images"):
        build_datasets(folder_data_config(image_folder))


def test_grayscale_and_rgba_images_become_rgb(tmp_path: Path) -> None:
    for name, mode in [("gray", "L"), ("alpha", "RGBA")]:
        (tmp_path / name).mkdir()
        for i in range(3):
            Image.new(mode, (10, 10)).save(tmp_path / name / f"{i}.png")
    datasets = build_datasets(folder_data_config(tmp_path, val_fraction=0.0, test_fraction=0.0))
    assert all(datasets.train[i][0].shape == (3, 16, 16) for i in range(len(datasets.train)))


# --- shapes -----------------------------------------------------------------------


def test_shapes_are_deterministic_and_labelled_by_shape() -> None:
    renderer = ShapeRenderer(seed=0)
    assert renderer(5).tobytes() == ShapeRenderer(seed=0)(5).tobytes()
    assert renderer(5).tobytes() != ShapeRenderer(seed=1)(5).tobytes()
    datasets = build_datasets(DataConfig(dataset="shapes", shapes_per_class=20, image_size=32))
    assert datasets.class_names == list(SHAPE_CLASSES)
    assert Counter(datasets.train.labels) == {c: 16 for c in range(6)}
    assert datasets.train[0][0].shape == (3, 32, 32)


def test_every_shape_kind_is_drawn() -> None:
    renderer = ShapeRenderer(seed=0)
    for index in range(len(SHAPE_CLASSES)):
        image = renderer(index)
        assert len(image.getcolors(maxcolors=10_000)) > 1, f"{SHAPE_CLASSES[index]} drew nothing"


# --- CIFAR-10 (torchvision dataset replaced by a fake: no download) ----------------


class FakeCIFAR10:
    classes = [f"class{i}" for i in range(10)]

    def __init__(self, root: str, train: bool, download: bool) -> None:
        n = 100 if train else 20
        self.targets = [i % 10 for i in range(n)]

    def __getitem__(self, index: int):
        return Image.new("RGB", (32, 32), (index % 256, 0, 0)), self.targets[index]


def test_cifar10_uses_its_official_test_set(monkeypatch: pytest.MonkeyPatch) -> None:
    import torchvision.datasets

    monkeypatch.setattr(torchvision.datasets, "CIFAR10", FakeCIFAR10)
    datasets = build_datasets(DataConfig(dataset="cifar10", image_size=32, val_fraction=0.1))
    assert datasets.class_names == FakeCIFAR10.classes
    assert (len(datasets.train), len(datasets.val), len(datasets.test)) == (90, 10, 20)
    assert datasets.test[0][0].shape == (3, 32, 32)


# --- transforms -------------------------------------------------------------------


@pytest.mark.parametrize("augment", ["none", "light", "strong"])
@pytest.mark.parametrize("train", [True, False])
def test_transforms_output_the_model_input_size(augment: str, train: bool) -> None:
    transform = build_transform(DataConfig(image_size=24, augment=augment), train=train)
    assert transform(Image.new("RGB", (50, 30))).shape == (3, 24, 24)


def test_eval_transform_is_deterministic_and_denormalize_inverts_it() -> None:
    config = DataConfig(image_size=8, augment="strong")
    image = Image.new("RGB", (8, 8), (255, 128, 0))
    transform = build_transform(config, train=False)
    torch.testing.assert_close(transform(image), transform(image))
    restored = denormalize(transform(image), config)
    torch.testing.assert_close(restored[:, 0, 0], torch.tensor([1.0, 128 / 255, 0.0]))


def test_images_are_closed_after_loading(image_folder: Path) -> None:
    path = next((image_folder / "red").iterdir())
    assert data.open_image(path).size == (20, 20)  # usable after the file is closed
