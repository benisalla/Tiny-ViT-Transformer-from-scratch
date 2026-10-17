# Legacy code (kept for reference)

`tiny_vit_transformer_from_scratch/` is the first version of this project,
unchanged except that cached `__pycache__` files and the 4.5 MB app background
image were removed. Its `pyproject.toml` / `poetry.lock` are here too. The
maintained code is in [`src/vit_classifier`](../src/vit_classifier). This page lists what
was wrong with the first version and where each problem is fixed.

Checkpoints of the first version cannot be loaded by the new code (module
names changed, and their `model_args` described a different MLP width than
the weights actually had, see the model table below). None were published.

## The blocking problems

1. **`train.py` could not run.** It called `VisionTransformer(vitconfig)` but
   the constructor takes ten keyword arguments, not a config object, so it
   crashed on the first line. Its imports (`from core.config import ...`) only
   worked from inside the package folder, while the model files imported
   `tiny_vit_transformer_from_scratch.model...`: no working directory satisfied
   both. **Fix:** one package, `vit-train` command, tested end to end.
2. **`save_checkpoints()` crashed.** It required an `epoch` argument that
   `train.py` never passed. **Fix:** `save_checkpoint()` saves model, optimizer,
   scheduler, RNG state, configs and class names, atomically.
3. **Dropout stayed on during evaluation.** `ViTSelfAttention.forward(x,
   training=True)` did `self.training = training`, overriding `model.eval()`,
   so validation, test and the app all ran with attention dropout whenever
   `d_rate > 0`. **Fix:** the module reads PyTorch's own `self.training`
   (tested: `test_dropout_is_active_in_train_mode_only`).
4. **The training schedule was unusable.** `num_epochs = total_iters *
   batch_size // train_size` = 3 000 000 × 16 / 2 000 = **24 000 epochs**, with
   a cosine schedule over epochs and `clip_grad_norm_(..., 0.05)` (a max norm
   20× smaller than usual, which turns every step into a very small fixed-length
   step). **Fix:** epochs you choose, per-step warmup + cosine decay, clip at 1.0.
5. **The app could not start anywhere but one PC.** It loaded the checkpoint
   from `C:\Users\Omar\Desktop\...`, read CSS and images by paths relative to
   the repo root, and assumed 256 px images while the config trained at
   128 px. **Fix:** `app/streamlit_app.py` takes `--checkpoint`, and every
   preprocessing setting comes from the checkpoint itself.

## All issues found

### Model

| First version | Problem | Fix |
|---|---|---|
| `ViTFeedForwardMLP` hard-coded `4 * n_embd` ("by mistake the model train with...") | `h_dim` (6 × n_embd in the config) was silently ignored; saved `model_args` lied about the architecture | `mlp_ratio` is used as configured |
| `self.n_patch = im_size**2 / p_size**2` | a float, passed on as a size | `ModelConfig.num_patches` (int), image size must be a multiple of the patch size |
| non-flash branch registered a causal `tril` mask of size `n_patch` | never used, wrong size (no room for [CLS]); a causal mask makes no sense for images anyway | no mask; explicit and fused attention give the same output (tested) |
| `encoder.dropout` | created, never applied | embedding dropout applied after the position embeddings |
| `_init_weights` branch for `ViTConv2dEmbedding` | read `module.bias`, which does not exist: would crash with the conv embedding | one init routine for both embeddings (tested) |
| `cls_token` of the default embedding: `torch.randn` (std 1) | 50× larger than every other weight | std 0.02 |
| scaled init looked for `c_proj.weight` | did not reach the MLP output layer | scaled init on both residual branches (`out_proj`, `fc2`) |
| head initialised randomly | the first loss is not ln(C), so a broken setup is harder to spot | zero head: first loss is exactly ln(C) (tested) |
| `init_from_config()` | referenced `embd_dim` / `head_dim`, which the config does not have | removed; `VisionTransformer(ModelConfig)` |
| prints the parameter count in `__init__` | noise in every test and app load | `num_parameters()` |
| `ViTPyCon2DEmbedding` | a duplicate of `ViTConv2dEmbedding` | two documented variants: `--patch-embed linear` / `conv`, proven equivalent in `test_conv_and_linear_patch_embeddings_compute_the_same_map` |
| weight decay on every parameter | decays LayerNorms, biases, [CLS] and position embeddings | AdamW groups: decay only weight matrices |

### Training and evaluation

| First version | Problem | Fix |
|---|---|---|
| `live_plot_dual` with `IPython.display` inside `train.py` | a notebook helper in a script; IPython was a hard dependency | text logs; live plots in the notebook via a callback |
| checkpoint every 100 epochs only, no "best" | lose everything on a crash; no model selection | `last.pt` after every validation, `best.pt` on the best validation accuracy, exact resume |
| `torch.load` without `weights_only` | can execute code from a file | `weights_only=True` |
| no mixed precision, no multi-GPU | slow on modern GPUs | bf16 / fp16 autocast, DDP with `torchrun` |
| `finetune.py` | empty file | removed; `--resume` continues a run |
| `evaluate_visualize_model` | showed one batch and printed accuracy only | `vit-evaluate`: accuracy, per-class recall / precision, confusion matrix, example table |
| reported 81.6 % test accuracy | no script, data split or checkpoint to reproduce it | results in the README come with the exact command |

### Data

| First version | Problem | Fix |
|---|---|---|
| `os.listdir(class_dir)[:max_img_cls]` | the first N files in whatever order the filesystem returns: not reproducible | sorted files, seeded random subset |
| class names hard-coded in three places (config, app, README) with different spellings than the PlantVillage folders | the app could print the wrong label | class names read from the folders and stored in the checkpoint |
| `train_test_split` without `random_state` | a different test set on every run, so the test accuracy can include training images of an earlier run | stratified split from `split_seed` only |
| heavy augmentation always on (8 transforms incl. 90° rotation, AutoAugment, erasing) | slow, and too strong for a first sanity check | `--augment none / light / strong` |
| `data/utils.py` defined `show_one_image` twice | the second silently replaced the first | removed (plots live in the notebook) |
| `max_img_cls = 20`, `max_cls = 4` left in the default config | the "real" config trained on 80 images | full data by default; `--max-per-class` for quick runs |

### App

| First version | Problem | Fix |
|---|---|---|
| hard-coded Windows checkpoint path | runs on one PC only | `--checkpoint` flag and sidebar field, clear error if missing |
| `im_size = 256` in the app vs 128 in training | wrong input size | image size and normalisation from the checkpoint |
| text "can detect: 4 classes" next to a list of 12 | contradictory | class list from the checkpoint |
| 4.5 MB base64 background image injected on every rerun | slow page, hard to read | default Streamlit theme |
| `use_column_width` | deprecated Streamlit API | current API |
| no confidence, no explanation | only a label | top-k probabilities and an attention-rollout heatmap |
