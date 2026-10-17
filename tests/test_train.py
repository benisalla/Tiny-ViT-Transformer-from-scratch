"""The training loop: schedule, evaluation, checkpoints, resume, accumulation, CLI."""

import math
from pathlib import Path

import pytest
import torch
from conftest import folder_data_config, small_model_config, small_train_config
from torch.utils.data import DataLoader, TensorDataset

from vit_classifier.data import build_datasets
from vit_classifier.model import VisionTransformer
from vit_classifier.train import (
    BEST_CHECKPOINT,
    LAST_CHECKPOINT,
    build_dataloaders,
    evaluate,
    load_checkpoint,
    lr_lambda,
    main,
    resolve_precision,
    train,
)

# --- helpers ----------------------------------------------------------------------


def run(tmp_path: Path, image_folder: Path, name: str = "run", **overrides):
    """Train a small model on the colour folder; returns (model, state, train_config)."""
    torch.manual_seed(0)
    data_config = folder_data_config(image_folder)
    datasets = build_datasets(data_config)
    config = small_train_config(tmp_path / name, **overrides)
    model = VisionTransformer(small_model_config())
    train_loader, val_loader = build_dataloaders(
        datasets.train, datasets.val, config, torch.device("cpu")
    )
    state = train(model, train_loader, val_loader, data_config, config, datasets.class_names)
    return model, state, config


# --- schedule and precision -------------------------------------------------------


def test_lr_schedule_warms_up_then_decays_to_zero() -> None:
    values = [lr_lambda(step, warmup_steps=10, total_steps=100) for step in range(101)]
    assert values[0] == pytest.approx(0.1) and values[9] == pytest.approx(1.0)
    assert values[10] == pytest.approx(1.0)
    assert values[55] == pytest.approx(0.5)  # half-way through the cosine
    assert values[100] == pytest.approx(0.0, abs=1e-12)
    assert all(a >= b for a, b in zip(values[10:], values[11:], strict=False))


def test_precision_choice() -> None:
    cpu = torch.device("cpu")
    assert resolve_precision("auto", cpu) == (None, False)
    assert resolve_precision("bf16", cpu) == (torch.bfloat16, False)
    with pytest.raises(ValueError):
        resolve_precision("fp16", cpu)


# --- evaluation -------------------------------------------------------------------


class Constant(torch.nn.Module):
    """Always predicts class 1 with probability ~1."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([0.0, 50.0, 0.0]).expand(x.shape[0], 3)


def test_evaluate_counts_every_sample() -> None:
    labels = torch.tensor([1, 1, 1, 0, 2])  # 3 of 5 are class 1
    loader = DataLoader(TensorDataset(torch.zeros(5, 1), labels), batch_size=2)  # uneven last batch
    metrics = evaluate(Constant(), loader, torch.device("cpu"))
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["loss"] == pytest.approx(2 * 50 / 5, rel=1e-4)  # two samples cost 50 nats


# --- the loop ---------------------------------------------------------------------


def test_training_learns_and_writes_checkpoints(tmp_path: Path, image_folder: Path) -> None:
    events = []
    torch.manual_seed(0)
    data_config = folder_data_config(image_folder)
    datasets = build_datasets(data_config)
    config = small_train_config(tmp_path / "run", epochs=6)
    model = VisionTransformer(small_model_config())
    loaders = build_dataloaders(datasets.train, datasets.val, config, torch.device("cpu"))
    state = train(model, *loaders, data_config, config, datasets.class_names, events.append)

    assert state.step == 6 * (24 // 4)
    train_losses = [e["loss"] for e in events if e["kind"] == "train"]
    assert train_losses[0] == pytest.approx(math.log(3), abs=0.15)  # starts at chance
    assert train_losses[-1] < 0.3 * train_losses[0]
    assert state.best_val_accuracy == 1.0  # colours are easy
    assert {"loss", "accuracy", "lr", "grad_norm"} <= set(events[0])
    for name in (BEST_CHECKPOINT, LAST_CHECKPOINT):
        checkpoint = torch.load(tmp_path / "run" / name, weights_only=True)  # safe to load
        assert checkpoint["class_names"] == ["blue", "green", "red"]
        assert checkpoint["model_config"]["num_classes"] == 3


def test_resume_mid_training_matches_bit_for_bit(tmp_path: Path, image_folder: Path) -> None:
    """Stop after epoch 1 of a 3-epoch run (eval_every writes the checkpoint), resume, compare."""
    full, _, _ = run(tmp_path, image_folder, "full", epochs=3)
    # Run the same 3-epoch schedule but keep the checkpoint written at step 6 (end of epoch 1).
    torch.manual_seed(0)
    data_config = folder_data_config(image_folder)
    datasets = build_datasets(data_config)
    config = small_train_config(tmp_path / "partial", epochs=3)
    model = VisionTransformer(small_model_config())
    loaders = build_dataloaders(datasets.train, datasets.val, config, torch.device("cpu"))
    saved = {}

    def stop_after_first_epoch(event: dict) -> None:
        if event["kind"] == "val" and event["step"] == 6:
            saved["checkpoint"] = torch.load(
                tmp_path / "partial" / LAST_CHECKPOINT, weights_only=True
            )
            raise KeyboardInterrupt  # simulate a crash right after the save

    with pytest.raises(KeyboardInterrupt):
        train(model, *loaders, data_config, config, datasets.class_names, stop_after_first_epoch)
    torch.save(saved["checkpoint"], tmp_path / "crash.pt")

    resumed, state, _ = run(
        tmp_path, image_folder, "resumed", epochs=3, resume=str(tmp_path / "crash.pt")
    )
    assert state.step == 18
    for (name, a), b in zip(full.state_dict().items(), resumed.state_dict().values(), strict=True):
        torch.testing.assert_close(a, b, msg=f"{name} differs after resume")


def test_same_seed_same_run(tmp_path: Path, image_folder: Path) -> None:
    a, _, _ = run(tmp_path, image_folder, "a", epochs=1)
    b, _, _ = run(tmp_path, image_folder, "b", epochs=1)
    for x, y in zip(a.state_dict().values(), b.state_dict().values(), strict=True):
        torch.testing.assert_close(x, y)


def test_gradient_accumulation_matches_one_big_batch(tmp_path: Path, image_folder: Path) -> None:
    """2 micro-batches of 4 + accumulation == 1 batch of 8 (same images, same order)."""
    big, _, _ = run(tmp_path, image_folder, "big", epochs=1, batch_size=8)
    small, _, _ = run(tmp_path, image_folder, "small", epochs=1, batch_size=4, grad_accum_steps=2)
    for (name, x), y in zip(big.state_dict().items(), small.state_dict().values(), strict=True):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4, msg=name)


def test_too_small_train_set_is_reported(tmp_path: Path, image_folder: Path) -> None:
    with pytest.raises(ValueError, match="too small"):
        run(tmp_path, image_folder, batch_size=16, grad_accum_steps=2)


# --- command line -----------------------------------------------------------------


def test_cli_trains_on_synthetic_shapes(tmp_path: Path) -> None:
    out = tmp_path / "cli"
    main([
        "--dataset", "shapes", "--shapes-per-class", "8", "--image-size", "16",
        "--patch-size", "4", "--d-model", "16", "--n-layers", "1", "--n-heads", "2",
        "--epochs", "1", "--batch-size", "8", "--num-workers", "0", "--device", "cpu",
        "--output-dir", str(out),
    ])  # fmt: skip
    checkpoint = torch.load(out / LAST_CHECKPOINT, weights_only=True)
    assert checkpoint["model_config"]["d_model"] == 16
    assert checkpoint["data_config"]["dataset"] == "shapes"
    assert len(checkpoint["class_names"]) == 6


def test_cli_overfit_sanity_check_reaches_100_percent(
    tmp_path: Path, image_folder: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    out = tmp_path / "overfit"
    main([
        "--dataset", "folder", "--data-dir", str(image_folder), "--overfit", "6",
        "--image-size", "16", "--patch-size", "4", "--d-model", "32", "--n-layers", "2",
        "--n-heads", "4", "--epochs", "40", "--lr", "3e-3", "--warmup-steps", "5",
        "--num-workers", "0", "--device", "cpu", "--output-dir", str(out),
    ])  # fmt: skip
    checkpoint = torch.load(out / BEST_CHECKPOINT, weights_only=True)
    assert checkpoint["best_val_accuracy"] == 1.0
    # overfit mode switches off every regulariser
    assert checkpoint["model_config"]["dropout"] == 0.0
    assert checkpoint["data_config"]["augment"] == "none"
    assert checkpoint["train_config"]["batch_size"] == 6
    # 100% must not be trivial: the 6 images cover all 3 classes (2 each)
    assert "overfit on 6 images: {'blue': 2, 'green': 2, 'red': 2}" in caplog.text


def test_load_checkpoint_restores_counters(tmp_path: Path, image_folder: Path) -> None:
    model, state, _ = run(tmp_path, image_folder, epochs=1)
    fresh = VisionTransformer(small_model_config())
    restored = load_checkpoint(tmp_path / "run" / LAST_CHECKPOINT, fresh)
    assert restored.step == state.step
    assert restored.history == state.history
    for x, y in zip(model.state_dict().values(), fresh.state_dict().values(), strict=True):
        torch.testing.assert_close(x, y)


def test_help_shows_defaults_but_not_none(capsys: pytest.CaptureFixture) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert "(default: shapes)" in text and "(default: 0.0005)" in text
    assert "default: None" not in text
