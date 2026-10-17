# Testing

Two kinds of evidence: a unit-test suite (fast, CPU, offline) that checks the
code does the right maths, and real training runs that check the model
actually learns.

## 1. The test suite

```bash
pip install -e '.[dev]'
pytest                                   # all tests, CPU only, no network
pytest --cov=tiny_vit                    # with line coverage (96 %)
pytest tests/test_model.py -v            # one file, verbose
ruff check . && ruff format --check .    # lint
```

| File | What it proves |
|---|---|
| `test_model.py` | the ViT itself, see the sanity checks below |
| `test_data.py` | splits are disjoint, complete, class-balanced and depend only on the seed; tiny classes keep a training image; folder labels really are the folders; class subsets and caps; clear errors for missing folders / classes / images; grayscale and RGBA become RGB; synthetic shapes are deterministic and every kind is drawn; CIFAR-10 uses its official test set (torchvision mocked: no download); every augmentation level outputs the model's input size; `denormalize` inverts the transform |
| `test_train.py` | warmup + cosine schedule; precision choice; evaluation counts every sample (uneven last batch); a short run learns (first loss = ln 3, final loss < 30 % of it, 100 % validation accuracy) and writes safe checkpoints; **a crashed and resumed run ends bit-for-bit equal to an uninterrupted one**; same seed = same run; **gradient accumulation (2 × 4) = one batch of 8**; the CLI trains on shapes; `--overfit` reaches 100 % and turns regularisation off; `--help` |
| `test_predict.py` | a checkpoint rebuilds the identical model; predictions are sorted probabilities summing to 1; batch and single predictions agree; heatmap shape / range and overlay; `vit-predict` text and JSON output; recall / precision maths; `vit-evaluate` on the held-out split with report and example table; refuses data whose classes differ from the checkpoint's |
| `test_distributed.py` | 2 processes (gloo, like `torchrun` on 2 GPUs) with different random inits end with the same step count and identical weights; only rank 0 writes checkpoints |
| `test_app_and_notebook.py` | the Streamlit app classifies and draws the attention overlay; the page renders, reports a missing checkpoint, and loads a real one (Streamlit `AppTest`); the notebook embeds the current `src/` code verbatim, the committed notebook is up to date, and every code cell is valid Python |
| `test_config.py` | impossible settings are rejected (image size not a multiple of the patch, width not divisible by heads, ...); configs round-trip through a `weights_only` checkpoint; every preset is valid |

### Neural-network sanity checks (`test_model.py`)

These catch the bugs that do not crash but silently make a model train badly:

| Check | Why it matters |
|---|---|
| initial loss = ln(num_classes) exactly | the head starts at zero; any other value means labels or loss are wired wrong |
| every parameter receives a non-zero gradient and is updated | finds layers that are created but skipped in `forward`, or cut off from the loss |
| samples of a batch do not influence each other | no accidental mixing across the batch dimension (e.g. a wrong `reshape`) |
| swapping image halves changes the output | position embeddings are used; without them a ViT sees a bag of patches |
| dropout is active in `train()` and off in `eval()` | the first version left it on at evaluation |
| explicit attention = fused `scaled_dot_product_attention`; attention rows sum to 1 | the visualisation path computes the same thing as the training path |
| linear and conv patch embeddings compute the same map | the two `--patch-embed` options are really equivalent |
| `patchify` cuts row-major patches | patch n is where the heatmap draws it |
| gradients match finite differences (`gradcheck`, float64) | backprop through the whole network is correct |
| residual stream std stays within 3× over 12 blocks | the scaled init keeps deep models stable |
| AdamW groups cover every parameter once; no decay on norms, biases, [CLS], positions | weight decay goes where it should |
| the model memorises one batch (loss < 0.05) | it can fit at all |
| bf16 autocast output is close to fp32 | mixed precision is safe |
| attention rollout is a distribution over patches, with no NaN in edge cases | the app's heatmap is well defined |

The notebook is also executed end to end in its `SMOKE_TEST` mode (shapes)
with `jupyter nbconvert --execute`.

## 2. Real training runs

All runs on a laptop CPU (8 cores, no GPU) **shared with other heavy jobs**,
so the times are pessimistic. Every number is reproducible with the commands
shown (same seeds, same splits).

### 2.1 Overfit sanity check

```bash
vit-train --dataset shapes --overfit 32 --epochs 200 --lr 1e-3 --warmup-steps 10 \
          --log-every 10 --output-dir checkpoints/overfit
```

The 32 images are balanced over the classes (the log prints
`overfit on 32 images: {'circle': 6, 'cross': 5, 'diamond': 5, 'ring': 5, 'square': 6, 'triangle': 5}`).
The first loss is 1.790 (ln 6 = 1.792). Validation accuracy (on the same 32
images) reached **100 % at step 120** and stayed there; loss 0.0055 at step
300. With only 100 steps it stops at 96.9 %, hence 200 epochs above.

> An earlier version of `--overfit` took the first N training images, which
> are all of one class (splits are grouped by class), and "reached 100 %" at
> step 1. Caught by this run, fixed with a class-balanced subset, and now
> guarded by `test_cli_overfit_sanity_check_reaches_100_percent`.

### 2.2 Synthetic shapes: the model learns from pixels

```bash
vit-train --dataset shapes --epochs 15 --lr 1e-3 --warmup-steps 200 --output-dir checkpoints/shapes
vit-evaluate --checkpoint checkpoints/shapes/best.pt --examples docs/examples/shapes --num-examples 10
```

0.81 M parameters, 9 600 training images. The run was stopped at step 850
and resumed from `last.pt` (`--resume`), so resume was exercised on a real
run too. Validation accuracy per epoch:

| epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 12 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| val acc | 19.4 % | 29.7 % | 41.5 % | 62.4 % | 82.6 % | 79.9 % | 88.5 % | 90.6 % | 94.6 % | 96.5 % | 97.2 % |

**Test: 97.33 % on 1 200 unseen images** (chance: 16.7 %).

| class | recall | precision |
|---|---:|---:|
| circle | 98.5 % | 91.2 % |
| square | 99.0 % | 99.5 % |
| triangle | 100.0 % | 100.0 % |
| diamond | 96.0 % | 99.0 % |
| cross | 93.0 % | 97.9 % |
| ring | 97.5 % | 97.0 % |

Most errors are crosses and diamonds predicted as circles (small shapes
shrunk to 32 px lose their corners). Real test outputs:
[docs/examples/shapes](examples/shapes/README.md).

### 2.3 PlantVillage, 12 tomato / potato classes

```bash
vit-train --dataset folder --data-dir data/plantvillage_hf/color \
          --image-size 64 --patch-size 8 --d-model 128 --n-layers 4 --n-heads 4 \
          --augment light --epochs 12 --batch-size 64 --lr 1e-3 --warmup-steps 300 \
          --output-dir checkpoints/plantvillage
vit-evaluate --checkpoint checkpoints/plantvillage/best.pt --examples docs/examples/plantvillage
```

A CPU-sized model (0.83 M parameters, 64 px). 11 965 train / 1 495 val /
1 495 test images; 66 minutes. Validation accuracy rose every epoch
(34.9 % → 54.9 % → 60.3 % → 71.4 % → … → 84.2 % → 84.6 %), so longer
training and the full-size model should do better.

**Test: 85.89% on 1495 unseen leaves** (chance: 8.3 %; largest class: 14 %).

| class | recall | precision | test images |
|---|---:|---:|---:|
| Potato · Early blight | 79.0% | 95.2% | 100 |
| Potato · Late blight | 73.0% | 83.9% | 100 |
| Potato · healthy | 66.7% | 90.9% | 15 |
| Tomato · Bacterial spot | 95.3% | 95.3% | 213 |
| Tomato · Early blight | 70.0% | 78.7% | 100 |
| Tomato · Late blight | 84.8% | 88.5% | 191 |
| Tomato · Leaf Mold | 92.6% | 76.5% | 95 |
| Tomato · Septoria leaf spot | 87.0% | 80.2% | 177 |
| Tomato · Spider mites Two-spotted spider mite | 92.9% | 81.2% | 168 |
| Tomato · Target Spot | 74.3% | 86.0% | 140 |
| Tomato · Tomato mosaic virus | 91.9% | 69.4% | 37 |
| Tomato · healthy | 95.0% | 94.4% | 159 |

Caveat: the attention rollout often highlights the background, not the
lesions (see the README figure). PlantVillage classes were photographed in
separate sessions, so the background is partly predictive of the label;
this accuracy may overstate how well the model would do on field photos.
