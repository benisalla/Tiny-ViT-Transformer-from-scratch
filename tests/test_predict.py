"""Inference (vit-predict) and evaluation (vit-evaluate) from a saved checkpoint."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import COLORS, write_checkpoint
from PIL import Image

from vit_classifier.evaluate import format_report, summarize
from vit_classifier.evaluate import main as evaluate_main
from vit_classifier.predict import Predictor, load_model, overlay_heatmap
from vit_classifier.predict import main as predict_main


def test_load_model_rebuilds_the_saved_model(tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path)
    model, data_config, class_names = load_model(path)
    assert not model.training  # ready for inference
    assert class_names == sorted(COLORS)
    assert data_config.image_size == model.config.image_size == 16
    saved = torch.load(path, weights_only=True)["model"]
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(weight, saved[name])


def test_predict_returns_sorted_probabilities(tmp_path: Path) -> None:
    predictor = Predictor.from_checkpoint(write_checkpoint(tmp_path), device="cpu")
    torch.nn.init.normal_(predictor.model.head.weight)  # non-uniform outputs
    ranked = predictor.predict(Image.new("RGB", (40, 30), "white"), top_k=10)
    assert len(ranked) == 3  # capped at the number of classes
    probs = [p for _, p in ranked]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) == pytest.approx(1.0, abs=1e-5)
    assert {name for name, _ in ranked} == set(COLORS)


def test_batch_and_single_predictions_agree(tmp_path: Path) -> None:
    predictor = Predictor.from_checkpoint(write_checkpoint(tmp_path), device="cpu")
    torch.nn.init.normal_(predictor.model.head.weight)
    a, b = Image.new("RGB", (16, 16), "red"), Image.new("RGB", (16, 16), "blue")
    both = predictor.probabilities([a, b])
    torch.testing.assert_close(both[1], predictor.probabilities([b])[0], atol=1e-6, rtol=1e-5)


def test_attention_heatmap_and_overlay(tmp_path: Path) -> None:
    predictor = Predictor.from_checkpoint(write_checkpoint(tmp_path), device="cpu")
    image = Image.new("RGB", (64, 48), "gray")
    heat = predictor.attention_heatmap(image)
    assert heat.shape == (4, 4)  # 16 px / 4 px patches
    assert heat.min() >= 0 and heat.max() <= 1
    overlay = overlay_heatmap(image, heat)
    assert overlay.size == image.size and overlay.mode == "RGB"
    full = overlay_heatmap(image, np.ones((4, 4)), alpha=1.0)
    assert full.getpixel((0, 0)) == (255, 0, 0)  # full attention = fully red


def test_predict_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = write_checkpoint(tmp_path)
    image = tmp_path / "x.png"
    Image.new("RGB", (16, 16), "green").save(image)
    predict_main(["--checkpoint", str(path), "--image", str(image), "--device", "cpu"])
    text = capsys.readouterr().out
    assert str(image) in text and "%" in text
    predict_main(["--checkpoint", str(path), "--image", str(image), "--json", "--top-k", "2"])
    result = json.loads(capsys.readouterr().out)
    assert len(result[str(image)]) == 2


# --- evaluation -------------------------------------------------------------------


def test_summarize_computes_recall_and_precision() -> None:
    # rows = truth, columns = prediction
    matrix = torch.tensor([[8, 2], [1, 9]])
    report = summarize(matrix, ["cat", "dog"])
    assert report["accuracy"] == pytest.approx(17 / 20)
    assert report["per_class"]["cat"]["recall"] == pytest.approx(0.8)
    assert report["per_class"]["cat"]["precision"] == pytest.approx(8 / 9)
    assert report["per_class"]["dog"]["support"] == 10
    assert "85.00%" in format_report(report, ["cat", "dog"])


def test_evaluate_cli_on_the_held_out_split(
    tmp_path: Path, image_folder: Path, capsys: pytest.CaptureFixture
) -> None:
    path = write_checkpoint(tmp_path / "run", data_dir=image_folder)
    examples = tmp_path / "examples"
    report_path = tmp_path / "report.json"
    evaluate_main([
        "--checkpoint", str(path), "--device", "cpu", "--num-workers", "0",
        "--report", str(report_path), "--examples", str(examples), "--num-examples", "3",
    ])  # fmt: skip
    report = json.loads(report_path.read_text())
    assert report["images"] == 3  # 10% of 30 images
    assert sum(map(sum, report["confusion_matrix"])) == 3
    assert "accuracy" in capsys.readouterr().out
    assert len(list(examples.glob("*.jpg"))) == 3
    assert (examples / "README.md").read_text().count("<img") == 3


def test_evaluate_refuses_data_with_other_classes(tmp_path: Path, image_folder: Path) -> None:
    path = write_checkpoint(tmp_path / "run", data_dir=image_folder)
    (image_folder / "yellow").mkdir()
    Image.new("RGB", (8, 8), "yellow").save(image_folder / "yellow" / "0.png")
    with pytest.raises(SystemExit, match="differ"):
        evaluate_main(["--checkpoint", str(path), "--device", "cpu", "--num-workers", "0"])
