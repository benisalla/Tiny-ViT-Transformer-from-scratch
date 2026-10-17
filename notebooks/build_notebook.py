"""Build notebooks/train_colab_kaggle.ipynb: a self-contained training notebook.

Every module of src/tiny_vit is embedded verbatim (as %%writefile cells), so the
notebook runs the exact package code and needs no GitHub access.
Regenerate after changing src/tiny_vit:
    python notebooks/build_notebook.py . notebooks/train_colab_kaggle.ipynb
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
MODULES = ["__init__", "config", "data", "model", "train", "predict", "evaluate"]

cells = []


def md(text: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    cells.append(nbf.v4.new_code_cell(text.strip()))


# ---------------------------------------------------------------------------
md("""
# Tiny-ViT: training notebook (Kaggle & Colab)

Train the from-scratch Vision Transformer with a **free Kaggle or Colab GPU**,
on synthetic shapes, CIFAR-10 or PlantVillage leaf diseases. The notebook is
**self-contained**: section 2 writes the `tiny_vit` package (identical to the
repo's `src/`), and training goes through the same `train()` function as the
`vit-train` command.

| | Kaggle | Colab |
|---|---|---|
| GPU | *Settings → Accelerator → GPU T4 x2* (or P100) | *Runtime → Change runtime type → T4 GPU* |
| Internet | *Settings → Internet → On* (to download data) | on by default |
| Checkpoints kept in | `/kaggle/working` (saved with "Save Version") | Google Drive if `USE_GOOGLE_DRIVE = True` |

**Sections:** 1. The package · 2. Settings · 3. Data · 4. Model ·
5. Training with a live dashboard · 6. Results · 7. Download
""")

# ---------------------------------------------------------------------------
md("""
## 1. The package

The cells below write `src/tiny_vit/*.py`, copied from the repo at build time.
Read them in this order: `config` → `data` → `model` → `train`.
""")
code("""
import os

os.makedirs("src/tiny_vit", exist_ok=True)
""")
for module in MODULES:
    source = (ROOT / "src" / "tiny_vit" / f"{module}.py").read_text()
    code(f"%%writefile src/tiny_vit/{module}.py\n{source}")
code("""
import sys

sys.path.insert(0, "src")
for name in [m for m in sys.modules if m.startswith("tiny_vit")]:
    del sys.modules[name]  # pick up the files just written if this cell is re-run
import tiny_vit.train as T

print("tiny_vit ready")
""")

# ---------------------------------------------------------------------------
md("""
## 2. Settings

| `DATASET` | Images | What for |
|---|---|---|
| `"shapes"` | 12 000 generated shapes, 6 classes, no download | prove the model learns from pixels (minutes) |
| `"cifar10"` | 60 000 photos 32×32, 10 classes, 170 MB | the standard small benchmark |
| `"plantvillage"` | ~15 000 leaf photos, 12 tomato / potato classes, ~0.3 GB | the original task of this project |

`SMOKE_TEST = True` runs a tiny version to check the whole notebook in a few
minutes. Keep it `False` for real training.
""")
code("""
import dataclasses
import subprocess
from pathlib import Path

import torch

from tiny_vit.config import DATASET_PRESETS, PLANTVILLAGE_TOMATO_POTATO, DataConfig, TrainConfig

SMOKE_TEST = False          # True = tiny quick check of the whole notebook
DATASET = "plantvillage"    # "shapes" | "cifar10" | "plantvillage"
USE_GOOGLE_DRIVE = False    # Colab only: keep checkpoints on Drive (survive disconnects)
RESUME = True               # continue from last.pt in OUTPUT_DIR if it exists

# --- where checkpoints go --------------------------------------------------------
ON_KAGGLE = Path("/kaggle/working").exists()
ON_COLAB = "COLAB_RELEASE_TAG" in os.environ
if ON_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
    OUTPUT_DIR = Path(f"/content/drive/MyDrive/tiny-vit/{DATASET}")
elif ON_KAGGLE:
    OUTPUT_DIR = Path(f"/kaggle/working/checkpoints/{DATASET}")
else:
    OUTPUT_DIR = Path(f"checkpoints/{DATASET}")

# --- data ----------------------------------------------------------------------
source = "folder" if DATASET == "plantvillage" else DATASET
if DATASET == "plantvillage":
    data_dir = "data/plantvillage_hf/color"
    if not Path(data_dir).exists():
        subprocess.check_call([
            "hf", "download", "leoho36/plant_village_dataset", "--repo-type", "dataset",
            "--include", "color/Tomato*", "--include", "color/Potato*",
            "--exclude", "*Yellow_Leaf_Curl*",  # not one of the 12 classes
            "--local-dir", "data/plantvillage_hf", "--max-workers", "16",
        ])
    classes = PLANTVILLAGE_TOMATO_POTATO
else:
    data_dir, classes = f"data/{DATASET}", None

preset = DATASET_PRESETS[source]
data_config = DataConfig(
    dataset=source, data_dir=data_dir, classes=classes, image_size=preset["image_size"],
    augment="strong" if DATASET != "shapes" else "light",
)
model_size = {k: preset[k] for k in ("image_size", "patch_size", "d_model", "n_layers", "n_heads")}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    props = torch.cuda.get_device_properties(0)
    precision = "bf16" if props.major >= 8 else "fp16"  # native bf16 needs compute capability 8+
    print(f"GPU: {props.name} ({props.total_memory / 1e9:.0f} GB)")
else:
    precision = "fp32"
    print("No GPU: enable one in the notebook settings (fine only for SMOKE_TEST / shapes).")

epochs = {"shapes": 15, "cifar10": 100, "plantvillage": 60}[DATASET]
train_config = TrainConfig(
    output_dir=str(OUTPUT_DIR), epochs=epochs, batch_size=128, lr=1e-3, warmup_steps=500,
    precision=precision, device=device.type, num_workers=min(4, os.cpu_count() or 1),
    log_every=25,
)
if SMOKE_TEST:
    OUTPUT_DIR = OUTPUT_DIR.parent / "smoke-test"  # never mixed with a real run
    data_config = dataclasses.replace(data_config, max_per_class=20, shapes_per_class=20)
    model_size.update(d_model=64, n_layers=2, n_heads=4)
    train_config = dataclasses.replace(
        train_config, output_dir=str(OUTPUT_DIR), epochs=3, batch_size=16, warmup_steps=2,
        log_every=2, num_workers=0,
    )
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if RESUME and (OUTPUT_DIR / T.LAST_CHECKPOINT).exists():
    train_config = dataclasses.replace(train_config, resume=str(OUTPUT_DIR / T.LAST_CHECKPOINT))

for name, cfg in [("DataConfig", data_config), ("TrainConfig", train_config)]:
    print(f"\\n{name}:")
    for key, value in dataclasses.asdict(cfg).items():
        print(f"  {key:18} {value}")
print(f"\\nmodel: {model_size}")
""")

# ---------------------------------------------------------------------------
md("""
## 3. Data

Stratified train / validation / test split (same class balance in each). The
test split is never used for training or for picking the best checkpoint.
""")
code("""
import logging

import matplotlib.pyplot as plt

from tiny_vit.data import build_datasets, denormalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                    force=True)
T.seed_everything(train_config.seed)
datasets = build_datasets(data_config)
class_names = datasets.class_names
print(f"{len(class_names)} classes | train {len(datasets.train):,} | "
      f"val {len(datasets.val):,} | test {len(datasets.test):,}")

fig, axes = plt.subplots(1, 6, figsize=(16, 3))
for ax, i in zip(axes, range(0, len(datasets.val), max(1, len(datasets.val) // 6)), strict=False):
    pixels, label = datasets.val[i]
    ax.imshow(denormalize(pixels, data_config).permute(1, 2, 0))
    ax.set_title(class_names[label], fontsize=8)
    ax.axis("off")
plt.show()

train_loader, val_loader = T.build_dataloaders(datasets.train, datasets.val, train_config, device)
print(f"{len(train_loader):,} train batches per epoch")
""")

# ---------------------------------------------------------------------------
md("## 4. Model")
code("""
import math

import torch.nn.functional as F

from tiny_vit.config import ModelConfig
from tiny_vit.model import VisionTransformer

model_config = ModelConfig(num_classes=len(class_names), **model_size)
model = VisionTransformer(model_config).to(device)
print(f"parameters: {model.num_parameters() / 1e6:.2f}M | tokens per image: "
      f"{model_config.num_patches} patches + [CLS]")

# Sanity check: the untrained model guesses uniformly, loss = ln(num_classes).
images, labels = next(iter(val_loader))
with torch.no_grad():
    loss = F.cross_entropy(model(images.to(device)), labels.to(device))
print(f"initial loss {loss.item():.3f} (uniform guessing = {math.log(len(class_names)):.3f})")
""")

# ---------------------------------------------------------------------------
md("""
## 5. Training with a live dashboard

Same loop as `vit-train`: AdamW, warmup + cosine schedule, mixed precision,
gradient clipping, label smoothing. After every epoch it validates and saves
`last.pt`, plus `best.pt` when the validation accuracy improves.

**Interrupted, or hit the session limit?** Run the notebook again with
`RESUME = True`: it continues from `last.pt` (same step, same data order).
""")
code("""
from IPython.display import clear_output

history = {"train": [], "val": []}


def draw_dashboard():
    train, val = history["train"], history["val"]
    clear_output(wait=True)
    fig, (ax_loss, ax_acc, ax_lr) = plt.subplots(1, 3, figsize=(16, 4))
    ax_loss.plot([r["step"] for r in train], [r["loss"] for r in train], label="train", alpha=0.8)
    ax_loss.plot([r["step"] for r in val], [r["loss"] for r in val], "o-", label="validation")
    ax_loss.set_title("loss")
    ax_acc.plot([r["step"] for r in train], [r["accuracy"] for r in train], label="train", alpha=0.8)
    ax_acc.plot([r["step"] for r in val], [r["accuracy"] for r in val], "o-", label="validation")
    ax_acc.set_title("accuracy")
    ax_lr.plot([r["step"] for r in train], [r["lr"] for r in train])
    ax_lr.set_title("learning rate")
    for ax in (ax_loss, ax_acc, ax_lr):
        ax.set_xlabel("optimizer step")
        ax.grid(alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
    plt.show()
    if train:
        last = train[-1]
        print(f"step {last['step']:,}/{last['total_steps']:,} | epoch {last['epoch'] + 1} | "
              f"loss {last['loss']:.4f} | {last['images_per_sec']:.0f} img/s")
    if val:
        print(f"validation accuracy {val[-1]['accuracy']:.2%} (best "
              f"{max(r['accuracy'] for r in val):.2%})")


def on_metrics(event):
    history[event["kind"]].append(event)
    draw_dashboard()


state = T.train(model, train_loader, val_loader, data_config, train_config, class_names,
                callback=on_metrics)
print(f"Training finished. Best validation accuracy {state.best_val_accuracy:.2%}")
""")

# ---------------------------------------------------------------------------
md("""
## 6. Results on the test split

The best checkpoint on images it never saw: accuracy, per-class recall /
precision, confusion matrix, and a table of real examples (`examples/README.md`)
ready to commit to GitHub.
""")
code("""
from tiny_vit import evaluate

evaluate.main(["--checkpoint", str(OUTPUT_DIR / T.BEST_CHECKPOINT), "--split", "test",
               "--report", str(OUTPUT_DIR / "test_report.json"),
               "--examples", str(OUTPUT_DIR / "examples"), "--num-examples", "12"])
""")
code("""
from PIL import Image

from tiny_vit.predict import Predictor, overlay_heatmap

predictor = Predictor.from_checkpoint(OUTPUT_DIR / T.BEST_CHECKPOINT, device.type)
fig, axes = plt.subplots(2, 6, figsize=(18, 6.5))
step = max(1, len(datasets.test) // 6)
for col, i in enumerate(range(0, len(datasets.test), step)[:6]):
    pixels, label = datasets.test[i]
    image = Image.fromarray((denormalize(pixels, data_config).permute(1, 2, 0) * 255).byte().numpy())
    name, prob = predictor.predict(image, top_k=1)[0]
    axes[0, col].imshow(image)
    axes[0, col].set_title(f"{name} ({prob:.0%})\\ntrue: {class_names[label]}", fontsize=7)
    axes[1, col].imshow(overlay_heatmap(image, predictor.attention_heatmap(image)))
    axes[1, col].set_title("attention rollout", fontsize=7)
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 7. Download the trained model

Zips `best.pt`, the test report and the examples (`last.pt` is left out: it is
only needed to resume). Use it with the repo:

```bash
unzip tiny-vit-run.zip -d checkpoints/run
vit-predict --checkpoint checkpoints/run/best.pt --image leaf.jpg
streamlit run app/streamlit_app.py -- --checkpoint checkpoints/run/best.pt
```
""")
code("""
import shutil
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    bundle = Path(tmp) / "bundle"
    bundle.mkdir()
    for name in [T.BEST_CHECKPOINT, "test_report.json"]:
        if (OUTPUT_DIR / name).exists():
            shutil.copy(OUTPUT_DIR / name, bundle / name)
    if (OUTPUT_DIR / "examples").exists():
        shutil.copytree(OUTPUT_DIR / "examples", bundle / "examples")
    archive = shutil.make_archive(str(OUTPUT_DIR.parent / "tiny-vit-run"), "zip", bundle)
print("created", archive)

if ON_COLAB:
    from google.colab import files

    files.download(archive)
elif ON_KAGGLE:
    print("Kaggle: click 'Save Version', then download it from the notebook's Output tab.")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "accelerator": "GPU",
    "colab": {"provenance": [], "gpuType": "T4"},
    "kaggle": {"accelerator": "gpu", "isInternetEnabled": True},
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
nbf.validate(nb)
nbf.write(nb, OUT)
print("wrote", OUT, len(cells), "cells")
