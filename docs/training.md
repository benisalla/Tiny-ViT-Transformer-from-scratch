# Training

`vit-train` (= `python -m tiny_vit.train`) trains a model and writes
checkpoints to `--output-dir`. Every flag and its default: `vit-train --help`.

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'      # package + tests + Streamlit + notebook tools
```

PyTorch: the command above installs the default wheel from PyPI (CUDA build on
Linux). For a smaller CPU-only install, first run
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## 2. Pick data and a model size

`--dataset shapes | cifar10 | folder` (see [datasets.md](datasets.md)). The
model sizes you do not set come from `config.DATASET_PRESETS`:

| `--dataset` | image | patch | tokens | `d_model` | layers | heads | parameters |
|---|---|---|---|---|---|---|---|
| `shapes` | 32 | 4 | 64 + 1 | 128 | 4 | 4 | 0.81 M |
| `cifar10` | 32 | 4 | 64 + 1 | 192 | 6 | 6 | 2.69 M |
| `folder` | 128 | 16 | 64 + 1 | 256 | 6 | 8 | 4.96 M |

Any of them can be overridden: `--image-size 224 --patch-size 16 --d-model 384
--n-layers 12 --n-heads 6` is ViT-S/16. The image size must be a multiple of
the patch size, and `d_model` a multiple of `n_heads`; both are checked.
Compute grows with the number of tokens squared (attention) and with
`d_model` squared (everything else).

## 3. First, the sanity checks

Before a long run, check the pipeline can learn at all:

```bash
# 1. memorise 32 images: train AND validate on them, no augmentation, no regularisation.
#    Validation accuracy must reach 100 %; if not, something is broken.
vit-train --dataset shapes --overfit 32 --epochs 200 --lr 1e-3 --warmup-steps 10 \
          --log-every 10 --output-dir checkpoints/overfit

# 2. learn a real (synthetic) task on a laptop CPU (under an hour on a free CPU)
vit-train --dataset shapes --epochs 15 --lr 1e-3 --warmup-steps 200 --output-dir checkpoints/shapes
```

What to look for in the log:

- **The first loss is `ln(num_classes)`** (1.79 for 6 classes, 2.48 for 12):
  the head starts at zero, so the model begins by guessing uniformly. A
  different value means the labels or the loss are wired wrong.
- `grad` (the gradient norm before clipping) stays in the 0.5-10 range. Values
  stuck at the clip (1.0) for long periods mean the learning rate is too high.
- Validation accuracy runs a few epochs behind training accuracy at first
  (training images are augmented, validation ones are not), then the gap
  should stay small. A train accuracy far above validation is overfitting:
  more augmentation (`--augment strong`), more dropout or more data.

## 4. One GPU

```bash
vit-train --dataset folder --data-dir data/plantvillage_hf/color --augment strong \
          --epochs 60 --batch-size 128 --lr 1e-3 --output-dir checkpoints/plantvillage
```

`--precision auto` (default) uses bf16 on Ampere and newer GPUs and fp16 with
loss scaling on older ones (T4, P100, V100). Use `--num-workers` ≈ the number
of CPU cores so image decoding keeps up with the GPU.

No GPU at hand? The [Kaggle / Colab notebook](../notebooks/train_colab_kaggle.ipynb)
runs the same code on a free one, with a live loss / accuracy dashboard.

## 5. Several GPUs or machines

The same command under `torchrun` uses
[DistributedDataParallel](https://pytorch.org/docs/stable/notes/ddp.html):
one process per GPU, each on a different slice of the data, gradients
averaged after each step.

```bash
# 4 GPUs on one machine
torchrun --nproc_per_node=4 -m tiny_vit.train --dataset cifar10 --batch-size 128 \
         --output-dir checkpoints/cifar10

# 2 machines x 8 GPUs (run on each machine, with its own --node_rank)
torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
         --rdzv_backend=c10d --rdzv_endpoint=HOST_OF_NODE_0:29500 \
         -m tiny_vit.train --dataset cifar10 --batch-size 128 --output-dir checkpoints/cifar10
```

`--batch-size` is per process: the effective batch is `batch_size ×
grad_accum_steps × number of GPUs` (printed at start). If you scale the batch
up a lot, scale `--lr` up too (roughly linearly up to batch ~1024) and keep
the warmup. Only rank 0 logs and writes checkpoints.

## 6. Resume

Every validation writes `last.pt` (model, optimizer, scheduler, RNG state,
step counter, configs); `best.pt` is written when the validation accuracy
improves. To continue an interrupted run, pass the same flags plus:

```bash
vit-train ... --resume checkpoints/run/last.pt
```

It continues from the same step with the same data order and the same random
stream: `test_resume_mid_training_matches_bit_for_bit` checks that a crashed
and resumed run ends with exactly the weights of an uninterrupted one.

## 7. Flags that matter most

| flag | default | notes |
|---|---|---|
| `--lr` | 5e-4 | peak learning rate after warmup. 1e-3 works well for the small presets with batch 64-128 |
| `--warmup-steps` | 500 | linear warmup; then cosine decay to 0 at the last step |
| `--epochs` | 30 | ViTs from scratch need many epochs on small datasets |
| `--augment` | light | `strong` for real photos trained from scratch |
| `--dropout` | 0.1 | on attention weights, residual branches and embeddings |
| `--label-smoothing` | 0.1 | 0 = plain cross-entropy |
| `--weight-decay` | 0.05 | AdamW, on weight matrices only |
| `--grad-clip` | 1.0 | gradient norm clipping, 0 = off |
| `--eval-every` | 0 | 0 = validate once per epoch; set a step count for long epochs |
| `--overfit N` | off | sanity check; also turns off augmentation, dropout, decay and smoothing |

## 8. Output

```
checkpoints/run/
├── best.pt     best validation accuracy: use this one for prediction
└── last.pt     latest state: use this one to resume
```

A checkpoint is a plain `dict` readable with `torch.load(path,
weights_only=True)`: `model` (state dict), `model_config`, `data_config`,
`train_config`, `class_names`, `history` (every logged metric), plus the
optimizer / scheduler / RNG states for resuming.
