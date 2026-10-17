# Datasets

`vit-train --dataset <name>` picks the source. All three go through the same
`build_datasets()` (in [`src/tiny_vit/data.py`](../src/tiny_vit/data.py)),
which returns train / validation / test sets and the class names. The class
names are stored in every checkpoint, so prediction never depends on a
hard-coded list.

| `--dataset` | Images | Classes | Download | Default model input | Use it to |
|---|---|---|---|---|---|
| `shapes` | 12 000 generated | 6 | none | 32 px, 4 px patches | prove the model learns from pixels, on a laptop CPU |
| `cifar10` | 60 000 photos, 32 × 32 | 10 | 170 MB, automatic | 32 px, 4 px patches | the standard small benchmark |
| `folder` | your own: `data_dir/<class>/<images>` | any | you provide it | 128 px, 16 px patches | real tasks, e.g. PlantVillage leaf diseases |

## Splits

- **shapes, folder**: a stratified split (each class keeps its proportions)
  into train / validation / test with `--val-fraction 0.1 --test-fraction 0.1`.
- **cifar10**: validation = a stratified 10 % of the official training set,
  test = the official 10 000-image test set.

The split depends **only on `--split-seed`** (default 0), not on the training
seed, so changing `--seed` changes initialisation and data order but never
moves test images into training. `vit-evaluate` rebuilds the same split from
the checkpoint.

## shapes: synthetic, no download

Six classes: circle, square, triangle, diamond, cross, ring. Each image has a
random colour (8 choices), size, position and a dark or light background, so
the **only** thing that determines the label is the shape. A model that scores
well has read the pixels. Image `i` is always drawn the same way (seeded by
`split_seed` and `i`), so runs are reproducible.

```bash
vit-train --dataset shapes --epochs 15 --lr 1e-3 --warmup-steps 200 --output-dir checkpoints/shapes
```

## cifar10

```bash
vit-train --dataset cifar10 --augment strong --epochs 100 --batch-size 128 --lr 1e-3 \
          --output-dir checkpoints/cifar10
```

Downloaded to `data/cifar10` on first use. ViTs trained from scratch on
CIFAR-10 typically reach ~80-90 % with strong augmentation and 100-200 epochs
(a GPU job: see the [notebook](../notebooks/train_colab_kaggle.ipynb)).

## PlantVillage (the original task of this project)

PlantVillage (Hughes & Salathé, 2015) is a set of ~54 000 photos of single
leaves on a plain background, 38 crop / disease classes. The first version of
this project used the 12 tomato and potato classes; they are listed in
`tiny_vit.config.PLANTVILLAGE_TOMATO_POTATO`.

Download those classes (~15 000 images, ~0.3 GB) from the Hugging Face mirror
[`leoho36/plant_village_dataset`](https://huggingface.co/datasets/leoho36/plant_village_dataset):

```bash
pip install -U huggingface_hub
hf download leoho36/plant_village_dataset --repo-type dataset \
   --include "color/Tomato*" --include "color/Potato*" --exclude "*Yellow_Leaf_Curl*" \
   --local-dir data/plantvillage_hf
```

This gives `data/plantvillage_hf/color/<class>/<image>.JPG`, the layout the
`folder` source expects. Train on it:

```bash
vit-train --dataset folder --data-dir data/plantvillage_hf/color --augment strong \
          --epochs 60 --batch-size 128 --lr 1e-3 --output-dir checkpoints/plantvillage
```

Notes:

- **Class imbalance**: from 152 (potato healthy) to 2 127 (tomato bacterial
  spot) images per class. `vit-evaluate` reports per-class recall so a weak
  minority class is visible, not hidden by the overall accuracy.
- **Easier than it looks**: PlantVillage photos are taken in the lab on a
  uniform background, several photos per leaf. Accuracy on it says little
  about photos taken in a field.
- Only some classes: `--classes "Tomato___healthy" "Tomato___Late_blight"`.
  Quick experiments: `--max-per-class 200` (a seeded random subset per class).

## Your own images

Any folder of class sub-folders works (jpg, jpeg, png, bmp, webp, tif):

```
data/my_dataset/
├── cat/   0001.jpg 0002.jpg ...
└── dog/   0001.jpg ...
```

```bash
vit-train --dataset folder --data-dir data/my_dataset --output-dir checkpoints/pets
```

Grayscale and RGBA images are converted to RGB. Class names are the folder
names, sorted.
