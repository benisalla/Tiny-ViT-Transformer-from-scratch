"""Score a checkpoint on its held-out test (or validation) split.

    vit-evaluate --checkpoint checkpoints/run/best.pt [--split test] [--examples docs/examples/run]

Prints overall accuracy, per-class recall / precision and the confusion matrix.
The split is rebuilt from the checkpoint's DataConfig (same split_seed), so it
contains exactly the images the model never trained on. With --examples, it
also writes thumbnails and a Markdown table (image, prediction, truth) of
images spread evenly over the split: real, not hand-picked, outputs.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image

from tiny_vit.data import build_datasets, denormalize
from tiny_vit.predict import Predictor


@torch.inference_mode()
def confusion_matrix(predictor: Predictor, loader: DataLoader) -> torch.Tensor:
    """(num_classes, num_classes) counts; row = true class, column = predicted class."""
    n = len(predictor.class_names)
    matrix = torch.zeros(n, n, dtype=torch.long)
    for images, labels in loader:
        predictions = predictor.model(images.to(predictor.device)).argmax(dim=-1).cpu()
        matrix += torch.bincount(labels * n + predictions, minlength=n * n).view(n, n)
    return matrix


def summarize(matrix: torch.Tensor, class_names: Sequence[str]) -> dict:
    """Accuracy plus per-class recall (how many of class c we found) and precision
    (how many predictions of c were right)."""
    correct = matrix.diag().float()
    per_class = {
        name: {
            "recall": float(correct[i] / max(int(matrix[i].sum()), 1)),
            "precision": float(correct[i] / max(int(matrix[:, i].sum()), 1)),
            "support": int(matrix[i].sum()),
        }
        for i, name in enumerate(class_names)
    }
    return {
        "accuracy": float(correct.sum() / max(int(matrix.sum()), 1)),
        "images": int(matrix.sum()),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def format_report(report: dict, class_names: Sequence[str]) -> str:
    width = max(len(n) for n in class_names)
    lines = [
        f"accuracy {report['accuracy']:.2%} on {report['images']} images",
        "",
        f"{'class':<{width}}  recall  precision  images",
    ]
    for name, row in report["per_class"].items():
        lines.append(
            f"{name:<{width}}  {row['recall']:6.1%}  {row['precision']:9.1%}  {row['support']:6d}"
        )
    lines += ["", "confusion matrix (row = true class, column = predicted):"]
    lines += ["  " + " ".join(f"{v:5d}" for v in row) for row in report["confusion_matrix"]]
    return "\n".join(lines)


def write_examples(predictor: Predictor, dataset, out_dir: Path, count: int) -> Path:
    """Thumbnails + README.md table for `count` images spread evenly over the dataset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    step = max(1, len(dataset) // count)
    rows = ["| image | prediction | confidence | truth | correct |", "|:---:|---|---:|---|:---:|"]
    for k, index in enumerate(range(0, len(dataset), step)[:count]):
        pixels, label = dataset[index]
        image = to_pil_image(denormalize(pixels, predictor.data_config))
        name, prob = predictor.predict(image, top_k=1)[0]
        truth = predictor.class_names[label]
        thumb = image.resize((96, 96), Image.NEAREST if image.width < 96 else Image.BICUBIC)
        thumb.save(out_dir / f"{k:02d}.jpg", quality=92)
        rows.append(
            f'| <img src="{k:02d}.jpg" width="72"> | {name} | {prob:.0%} | {truth} | '
            f"{'yes' if name == truth else '**no**'} |"
        )
    readme = out_dir / "README.md"
    readme.write_text("\n".join(rows) + "\n")
    return readme


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Evaluate a Tiny-ViT checkpoint on a held-out split.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--split", choices=("test", "val"), default="test")
    p.add_argument("--data-dir", help="override the dataset folder stored in the checkpoint")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--report", type=Path, help="also save the numbers as JSON")
    p.add_argument("--examples", type=Path, help="write an example table to this folder")
    p.add_argument("--num-examples", type=int, default=12)
    args = p.parse_args(argv)

    predictor = Predictor.from_checkpoint(args.checkpoint, args.device)
    if args.data_dir:
        predictor.data_config.data_dir = args.data_dir
    datasets = build_datasets(predictor.data_config)
    dataset = datasets.test if args.split == "test" else datasets.val
    if datasets.class_names != predictor.class_names:
        raise SystemExit(
            f"the data's classes {datasets.class_names} differ from the checkpoint's "
            f"{predictor.class_names}"
        )
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    report = summarize(confusion_matrix(predictor, loader), predictor.class_names)
    report["split"] = args.split
    print(format_report(report, predictor.class_names))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
    if args.examples:
        readme = write_examples(predictor, dataset, args.examples, args.num_examples)
        print(f"\nexamples written to {readme}")


if __name__ == "__main__":
    main()
