"""Classify images with a trained checkpoint.

    vit-predict --checkpoint checkpoints/run/best.pt --image leaf.jpg [more.jpg ...] [--top-k 3]

From Python:
    from vit_classifier.predict import Predictor
    predictor = Predictor.from_checkpoint("checkpoints/run/best.pt")
    predictor.predict(Image.open("leaf.jpg"))   # [("Tomato___healthy", 0.97), ...]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vit_classifier.config import DataConfig, ModelConfig, config_from_dict
from vit_classifier.data import build_transform
from vit_classifier.model import VisionTransformer, attention_rollout
from vit_classifier.train import resolve_device


def load_model(
    path: str | Path, device: torch.device | str = "cpu"
) -> tuple[VisionTransformer, DataConfig, list[str]]:
    """Rebuild a vit-train checkpoint: (model in eval mode, data config, class names)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = VisionTransformer(config_from_dict(ModelConfig, checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    data_config = config_from_dict(DataConfig, checkpoint["data_config"])
    return model, data_config, list(checkpoint["class_names"])


@dataclass
class Predictor:
    """A model plus everything needed to feed it raw images."""

    model: VisionTransformer
    data_config: DataConfig
    class_names: list[str]
    device: torch.device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "auto") -> Predictor:
        resolved = resolve_device(device)
        model, data_config, class_names = load_model(path, resolved)
        return cls(model, data_config, class_names, resolved)

    def preprocess(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """PIL images -> (B, 3, S, S) batch, exactly like the validation images."""
        transform = build_transform(self.data_config, train=False)
        return torch.stack([transform(image.convert("RGB")) for image in images]).to(self.device)

    @torch.inference_mode()
    def probabilities(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """(B, num_classes) class probabilities."""
        return self.model(self.preprocess(images)).float().softmax(dim=-1).cpu()

    def predict(self, image: Image.Image, top_k: int = 5) -> list[tuple[str, float]]:
        """[(class name, probability)], most likely first."""
        probs = self.probabilities([image])[0]
        values, indices = probs.topk(min(top_k, len(self.class_names)))
        return [
            (self.class_names[i], float(v)) for v, i in zip(values, indices.tolist(), strict=True)
        ]

    def attention_heatmap(self, image: Image.Image) -> np.ndarray:
        """Attention rollout as an (S/P, S/P) array in [0, 1]: where the model looked."""
        attentions = self.model.attention_maps(self.preprocess([image]))
        grid = self.data_config.image_size // self.model.config.patch_size
        heat = attention_rollout(attentions)[0].reshape(grid, grid).float().cpu().numpy()
        return (heat - heat.min()) / max(float(heat.max() - heat.min()), 1e-12)


def overlay_heatmap(image: Image.Image, heat: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend a heatmap (red = high attention) over the image, at the image's size."""
    image = image.convert("RGB")
    mask = Image.fromarray((heat * 255).astype(np.uint8)).resize(image.size, Image.BICUBIC)
    m = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    red = np.zeros_like(np.asarray(image, dtype=np.float32))
    red[..., 0] = 255.0
    pixels = np.asarray(image, dtype=np.float32)
    blended = pixels * (1 - alpha * m) + red * (alpha * m)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Classify images with a trained checkpoint.")
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--image", required=True, nargs="+", type=Path)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--json", action="store_true", help="print JSON instead of text")
    args = p.parse_args(argv)

    predictor = Predictor.from_checkpoint(args.checkpoint, args.device)
    results = {}
    for path in args.image:
        with Image.open(path) as image:
            results[str(path)] = predictor.predict(image, args.top_k)
    if args.json:
        print(
            json.dumps(
                {k: [{"class": c, "probability": p} for c, p in v] for k, v in results.items()},
                indent=2,
            )
        )
        return
    for path, ranked in results.items():
        print(path)
        for name, prob in ranked:
            print(f"  {prob:7.2%}  {name}")


if __name__ == "__main__":
    main()
