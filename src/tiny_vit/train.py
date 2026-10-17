"""Train the Vision Transformer.

The file is split into small pieces you can read top to bottom:

1. setup helpers   - seeding, device, mixed precision, multi-GPU (DDP) setup
2. data            - DataLoaders
3. optimisation    - AdamW + warmup/cosine learning-rate schedule
4. evaluation      - loss and accuracy on a held-out set
5. checkpoints     - save / load everything needed to resume (or to predict)
6. train()         - the training loop (takes ready-made objects, easy to test)
7. CLI             - main() builds everything from command-line flags

Run it with:  vit-train --help

Multi-GPU / multi-node: launch the same command with torchrun, e.g.
    torchrun --nproc_per_node=4 -m tiny_vit.train ...
torchrun starts one process per GPU and sets RANK / WORLD_SIZE / LOCAL_RANK;
the code below picks those up and wraps the model in DistributedDataParallel.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import time
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from tiny_vit.config import (
    DATASET_PRESETS,
    DATASETS,
    DataConfig,
    ModelConfig,
    TrainConfig,
    config_to_dict,
)
from tiny_vit.data import ImageDataset, balanced_subset, build_datasets, build_transform
from tiny_vit.model import VisionTransformer

logger = logging.getLogger(__name__)

LAST_CHECKPOINT = "last.pt"
BEST_CHECKPOINT = "best.pt"


# ---------------------------------------------------------------------------
# 1. Setup helpers
# ---------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    """Seed every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def resolve_device(name: str = "auto") -> torch.device:
    """Turn "auto" | "cpu" | "cuda" | "mps" into a torch.device."""
    # Under torchrun each process owns one GPU: cuda:LOCAL_RANK.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda", local_rank)
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_precision(precision: str, device: torch.device) -> tuple[torch.dtype | None, bool]:
    """Pick the autocast dtype and whether we need a GradScaler.

    Returns (dtype, use_scaler). dtype None means plain fp32 (no autocast).
    - bf16 has the same range as fp32, so it needs no loss scaling.
    - fp16 has a small range; small gradients underflow to 0 without a GradScaler.
    """
    if precision == "fp32":
        return None, False
    if precision == "bf16":
        return torch.bfloat16, False
    if precision == "fp16":
        if device.type != "cuda":
            raise ValueError("fp16 training needs a CUDA device")
        return torch.float16, True
    if precision == "auto":
        if device.type != "cuda":
            return None, False  # CPU / MPS: keep it simple and exact
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False
        return torch.float16, True
    raise ValueError(f"unknown precision: {precision!r}")


@dataclass(frozen=True)
class DistInfo:
    """Who am I in a multi-process run? Defaults describe a single process."""

    rank: int = 0  # global id of this process (0 .. world_size-1)
    world_size: int = 1  # total number of processes (= total GPUs)
    local_rank: int = 0  # id of this process on its own machine

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """Only the main process logs and writes checkpoints."""
        return self.rank == 0


def setup_distributed(device_name: str = "auto") -> DistInfo:
    """Join the process group if we were launched by torchrun; no-op otherwise."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return DistInfo()
    local_rank = int(os.environ["LOCAL_RANK"])
    use_cuda = device_name in ("auto", "cuda") and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)  # this process only talks to its own GPU
    # NCCL is the fast backend for GPUs; gloo works on CPU (handy for testing).
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank)


def get_dist_info() -> DistInfo:
    """Current process-group state (single process if none was started)."""
    if not (dist.is_available() and dist.is_initialized()):
        return DistInfo()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return DistInfo(dist.get_rank(), dist.get_world_size(), local_rank)


@contextmanager
def local_main_first(info: DistInfo) -> Iterator[None]:
    """Let local rank 0 run the block first (e.g. the CIFAR-10 download);
    the other processes on the machine wait, then reuse the files."""
    if info.enabled and info.local_rank != 0:
        dist.barrier()
    yield
    if info.enabled and info.local_rank == 0:
        dist.barrier()


def autocast(device: torch.device, dtype: torch.dtype | None) -> AbstractContextManager:
    """Mixed-precision context; does nothing when dtype is None."""
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None)


# ---------------------------------------------------------------------------
# 2. Data
# ---------------------------------------------------------------------------


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own python/numpy seed (torch is already seeded)."""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_dataloaders(
    train_dataset: Dataset, val_dataset: Dataset, config: TrainConfig, device: torch.device
) -> tuple[DataLoader, DataLoader]:
    """Create the train and validation DataLoaders.

    With several processes, a DistributedSampler gives each one a different
    slice of the data, so batch_size is *per process* (per GPU).
    """
    info = get_dist_info()
    workers = config.num_workers
    common = {
        "batch_size": config.batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",  # faster CPU -> GPU copies
        "persistent_workers": workers > 0,  # do not restart workers every epoch
        "worker_init_fn": seed_worker if workers > 0 else None,
    }
    generator = torch.Generator()
    generator.manual_seed(config.seed + info.rank)

    if info.enabled:
        train_sampler = DistributedSampler(
            train_dataset, shuffle=True, seed=config.seed, drop_last=True
        )
        # Note: to split evenly, the sampler may repeat a few val samples.
        val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)
        train_loader = DataLoader(
            train_dataset, sampler=train_sampler, drop_last=True, generator=generator, **common
        )
        val_loader = DataLoader(val_dataset, sampler=val_sampler, drop_last=False, **common)
        return train_loader, val_loader

    train_loader = DataLoader(
        train_dataset, shuffle=True, drop_last=True, generator=generator, **common
    )
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# 3. Optimisation
# ---------------------------------------------------------------------------


def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    """LR multiplier: linear warmup to 1, then cosine decay to 0.

    `step` is the number of optimizer steps already taken (starts at 0).
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps  # never exactly 0, so step 0 still learns
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, (step - warmup_steps) / decay_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_optimizer(model: VisionTransformer, config: TrainConfig, device: torch.device) -> AdamW:
    """AdamW; no weight decay on biases, norms, [CLS] and position embeddings."""
    groups = model.parameter_groups(config.weight_decay)
    # fused=True runs the whole update in one CUDA kernel (faster); CUDA only.
    return AdamW(groups, lr=config.lr, betas=(0.9, 0.999), fused=device.type == "cuda")


def build_scheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Warmup + cosine schedule. Call scheduler.step() once per optimizer step."""
    return LambdaLR(optimizer, lambda step: lr_lambda(step, warmup_steps, total_steps))


# ---------------------------------------------------------------------------
# 4. Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype | None = None,
) -> dict[str, float]:
    """Mean cross-entropy (no label smoothing) and top-1 accuracy over the loader.

    With several processes each one evaluates its slice; the sums are added up
    so every process gets the same, global numbers.
    """
    was_training = model.training
    model.eval()  # dropout off
    totals = torch.zeros(3, device=device)  # loss sum, correct, count
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with autocast(device, amp_dtype):
            logits = model(images)
        totals[0] += F.cross_entropy(logits.float(), labels, reduction="sum")
        totals[1] += (logits.argmax(dim=-1) == labels).sum()
        totals[2] += labels.numel()
    if get_dist_info().enabled:
        dist.all_reduce(totals)
    model.train(was_training)
    loss_sum, correct, count = totals.tolist()
    count = max(count, 1.0)
    return {"loss": loss_sum / count, "accuracy": correct / count}


# ---------------------------------------------------------------------------
# 5. Checkpoints
# ---------------------------------------------------------------------------


@dataclass
class TrainState:
    """Counters that must survive a restart."""

    step: int = 0  # optimizer steps done
    epoch: int = 0  # epoch we are in
    best_val_accuracy: float = -1.0
    history: list[dict[str, Any]] = field(default_factory=list)  # every logged event


def rng_state() -> dict[str, Any]:
    """Snapshot every RNG so a resumed run continues the same random stream."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        # plain lists so torch.load(weights_only=True) can read them back
        "numpy": (name, keys.tolist(), pos, has_gauss, cached),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore what rng_state() saved."""
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(
    path: Path,
    model: VisionTransformer,
    optimizer: Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    state: TrainState,
    data_config: DataConfig,
    train_config: TrainConfig,
    class_names: Sequence[str],
) -> None:
    """Save everything needed to resume training or to run the model. Written atomically."""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "step": state.step,
        "epoch": state.epoch,
        "best_val_accuracy": state.best_val_accuracy,
        "history": state.history,
        "class_names": list(class_names),
        "data_config": config_to_dict(data_config),
        "model_config": config_to_dict(model.config),
        "train_config": config_to_dict(train_config),
        "rng_state": rng_state(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, tmp_path)
    # os.replace is atomic: a crash mid-save never leaves a half-written file.
    os.replace(tmp_path, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LambdaLR | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> TrainState:
    """Load a checkpoint into the given objects and return the saved counters."""
    # weights_only=True refuses to run arbitrary pickled code (safe loading).
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and checkpoint["scaler"]:  # empty dict when scaler disabled
        scaler.load_state_dict(checkpoint["scaler"])
    set_rng_state(checkpoint["rng_state"])
    return TrainState(
        step=checkpoint["step"],
        epoch=checkpoint["epoch"],
        best_val_accuracy=checkpoint["best_val_accuracy"],
        history=list(checkpoint.get("history", [])),
    )


# ---------------------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------------------


def epoch_batches(loader: DataLoader, epoch: int, seed: int, skip: int) -> Iterator:
    """Batches of one epoch, in a shuffle order that depends only on (seed, epoch).

    Re-seeding per epoch means a resumed run sees the same order, so we can
    skip the batches that were already used before the restart.
    """
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)  # new shuffle each epoch, same on every process
    elif loader.generator is not None:
        loader.generator.manual_seed(seed + epoch)
    return islice(iter(loader), skip, None)


def train(
    model: VisionTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    data_config: DataConfig,
    train_config: TrainConfig,
    class_names: Sequence[str],
    callback: Callable[[dict[str, Any]], None] | None = None,
) -> TrainState:
    """Run the full training loop and return the final counters.

    Everything is passed in already built, so tests can use tiny datasets.
    `callback` (optional) receives every logged metric as a dict, on the main
    process only; the notebook uses it to draw live training curves:
        {"kind": "train", "step", "epoch", "total_steps", "loss", "accuracy", "lr",
         "grad_norm", "images_per_sec"}
        {"kind": "val", "step", "epoch", "loss", "accuracy"}
    """
    cfg = train_config
    info = get_dist_info()
    device = resolve_device(cfg.device)
    amp_dtype, use_scaler = resolve_precision(cfg.precision, device)
    model.to(device)
    model.train()
    raw_model = model  # the plain model: used for checkpoints and evaluation

    # One optimizer step = grad_accum_steps micro-batches.
    accum = cfg.grad_accum_steps
    steps_per_epoch = len(train_loader) // accum
    if steps_per_epoch == 0:
        raise ValueError("train set too small for batch_size * grad_accum_steps")
    total_steps = cfg.epochs * steps_per_epoch
    eval_every = cfg.eval_every or steps_per_epoch

    optimizer = build_optimizer(raw_model, cfg, device)
    scheduler = build_scheduler(optimizer, cfg.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = TrainState()
    if cfg.resume is not None:
        state = load_checkpoint(Path(cfg.resume), raw_model, optimizer, scheduler, scaler)
        logger.info("resumed from %s at step %d", cfg.resume, state.step)

    if info.enabled:
        # DDP averages the gradients of all processes after each backward pass.
        # It also copies rank 0's weights to everyone at start, so all begin equal.
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    def checkpoint(path: Path) -> None:
        if info.is_main:  # one writer is enough: all processes hold the same weights
            save_checkpoint(
                path, raw_model, optimizer, scheduler, scaler, state,
                data_config, cfg, class_names,
            )  # fmt: skip

    def emit(event: dict[str, Any]) -> None:
        if info.is_main:
            state.history.append(dict(event))
            if callback is not None:
                callback(event)

    def validate_and_save() -> None:
        metrics = evaluate(raw_model, val_loader, device, amp_dtype)
        logger.info(
            "val   | epoch %d | step %d | loss %.4f | acc %.4f",
            state.epoch, state.step, metrics["loss"], metrics["accuracy"],
        )  # fmt: skip
        if metrics["accuracy"] > state.best_val_accuracy:
            state.best_val_accuracy = metrics["accuracy"]
            checkpoint(output_dir / BEST_CHECKPOINT)
        checkpoint(output_dir / LAST_CHECKPOINT)
        # after the save: whoever receives this event can rely on last.pt being current
        emit({"kind": "val", "step": state.step, "epoch": state.epoch, **metrics})

    logger.info(
        "training %.2fM parameters: %d epochs x %d steps (effective batch %d = %d per process "
        "x %d accum x %d processes) on %s, amp=%s",
        raw_model.num_parameters() / 1e6, cfg.epochs, steps_per_epoch,
        cfg.batch_size * accum * info.world_size, cfg.batch_size, accum, info.world_size,
        device, amp_dtype,
    )  # fmt: skip

    # Running sums for the log line (kept on device: no GPU sync every step).
    window_loss = torch.zeros((), device=device)
    window_correct = torch.zeros((), device=device)
    window_images = 0
    window_steps = 0
    window_start = time.perf_counter()

    for epoch in range(state.step // steps_per_epoch, cfg.epochs):
        state.epoch = epoch
        done_in_epoch = state.step - epoch * steps_per_epoch  # > 0 only right after resume
        batches = epoch_batches(train_loader, epoch, cfg.seed, skip=done_in_epoch * accum)
        micro_batches = (steps_per_epoch - done_in_epoch) * accum  # drop the leftover tail

        for micro_step, (images, labels) in enumerate(islice(batches, micro_batches)):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            is_update_step = (micro_step + 1) % accum == 0
            # With DDP, skip the gradient all-reduce until the last micro-batch:
            # syncing once per optimizer step is enough and much cheaper.
            sync = model.no_sync() if info.enabled and not is_update_step else nullcontext()
            with sync:
                with autocast(device, amp_dtype):
                    logits = model(images)
                    loss = F.cross_entropy(
                        logits.float(), labels, label_smoothing=cfg.label_smoothing
                    )
                # Divide so the summed gradients equal the gradient of the mean loss.
                scaler.scale(loss / accum).backward()
            window_loss += loss.detach() / accum
            window_correct += (logits.detach().argmax(dim=-1) == labels).sum()
            window_images += labels.numel()

            if not is_update_step:
                continue  # keep accumulating gradients

            # --- one optimizer step ---
            scaler.unscale_(optimizer)  # back to real gradient values before clipping
            max_norm = cfg.grad_clip if cfg.grad_clip > 0 else math.inf
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)  # skips the step if fp16 gradients overflowed
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state.step += 1
            window_steps += 1

            if state.step % cfg.log_every == 0 or state.step == total_steps:
                elapsed = time.perf_counter() - window_start
                mean_loss = window_loss.item() / window_steps
                accuracy = window_correct.item() / max(window_images, 1)
                images_per_sec = window_images * info.world_size / elapsed
                lr = optimizer.param_groups[0]["lr"]
                logger.info(
                    "train | epoch %d | step %d/%d | loss %.4f | acc %.3f | lr %.2e | grad %.2f "
                    "| %.0f img/s",
                    epoch, state.step, total_steps, mean_loss, accuracy, lr,
                    grad_norm.item(), images_per_sec,
                )  # fmt: skip
                emit({
                    "kind": "train", "step": state.step, "epoch": epoch,
                    "total_steps": total_steps, "loss": mean_loss, "accuracy": accuracy,
                    "lr": lr, "grad_norm": grad_norm.item(), "images_per_sec": images_per_sec,
                })  # fmt: skip
                window_loss.zero_()
                window_correct.zero_()
                window_images = window_steps = 0
                window_start = time.perf_counter()

            if state.step % eval_every == 0 or state.step == total_steps:
                validate_and_save()
                window_start = time.perf_counter()  # don't count eval time in the speed

    state.epoch = cfg.epochs
    checkpoint(output_dir / LAST_CHECKPOINT)
    logger.info("done | best val accuracy %.4f", state.best_val_accuracy)
    return state


# ---------------------------------------------------------------------------
# 7. Command line
# ---------------------------------------------------------------------------

# Flags whose default depends on --dataset (see config.DATASET_PRESETS).
PRESET_FLAGS = ("image_size", "patch_size", "d_model", "n_layers", "n_heads")


class _HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Show "(default: X)" on every flag, except when there is no default."""

    def _get_help_string(self, action: argparse.Action) -> str:
        if action.default is None:
            return action.help or ""
        return super()._get_help_string(action) or ""


def build_parser() -> argparse.ArgumentParser:
    """All CLI flags. Defaults come from the config dataclasses."""
    d, t, m = DataConfig(), TrainConfig(), ModelConfig(num_classes=1)
    p = argparse.ArgumentParser(
        description="Train a Vision Transformer from scratch.", formatter_class=_HelpFormatter
    )
    g = p.add_argument_group("data")
    g.add_argument("--dataset", choices=DATASETS, default="shapes", help="image source")
    g.add_argument("--data-dir", help="dataset folder (default: data/<dataset>)")
    g.add_argument("--classes", nargs="+", help="folder dataset: only these class folders")
    g.add_argument("--max-per-class", type=int, help="folder dataset: cap images per class")
    g.add_argument("--val-fraction", type=float, default=d.val_fraction, help="held out per class")
    g.add_argument(
        "--test-fraction", type=float, default=d.test_fraction, help="held out per class"
    )
    g.add_argument("--split-seed", type=int, default=d.split_seed, help="seed of the split only")
    g.add_argument(
        "--augment", choices=("none", "light", "strong"), default=d.augment,
        help="training augmentation",
    )  # fmt: skip
    g.add_argument(
        "--shapes-per-class", type=int, default=d.shapes_per_class, help="shapes dataset size"
    )

    g = p.add_argument_group("model (unset sizes come from config.DATASET_PRESETS[--dataset])")
    g.add_argument("--image-size", type=int, help="input side in pixels")
    g.add_argument("--patch-size", type=int, help="patch side in pixels")
    g.add_argument("--d-model", type=int, help="token width")
    g.add_argument("--n-layers", type=int, help="Transformer blocks")
    g.add_argument("--n-heads", type=int, help="attention heads")
    g.add_argument("--mlp-ratio", type=float, default=m.mlp_ratio, help="MLP width / d_model")
    g.add_argument("--dropout", type=float, default=m.dropout, help="dropout rate")
    g.add_argument(
        "--patch-embed", choices=("linear", "conv"), default=m.patch_embed,
        help="how patches become tokens",
    )  # fmt: skip
    g.add_argument(
        "--pool", choices=("cls", "mean"), default=m.pool, help="what the head classifies"
    )

    g = p.add_argument_group("training")
    g.add_argument("--output-dir", default=t.output_dir, help="where checkpoints go")
    g.add_argument("--epochs", type=int, default=t.epochs, help="passes over the train set")
    g.add_argument("--batch-size", type=int, default=t.batch_size, help="per process / GPU")
    g.add_argument(
        "--grad-accum-steps", type=int, default=t.grad_accum_steps,
        help="micro-batches per optimizer step",
    )  # fmt: skip
    g.add_argument("--lr", type=float, default=t.lr, help="peak learning rate")
    g.add_argument("--weight-decay", type=float, default=t.weight_decay, help="AdamW decay")
    g.add_argument("--warmup-steps", type=int, default=t.warmup_steps, help="linear LR warmup")
    g.add_argument(
        "--label-smoothing", type=float, default=t.label_smoothing, help="0 = plain cross-entropy"
    )
    g.add_argument("--grad-clip", type=float, default=t.grad_clip, help="max grad norm, 0 = off")
    g.add_argument(
        "--precision", choices=("auto", "fp32", "bf16", "fp16"), default=t.precision,
        help="auto = bf16/fp16 on GPU, fp32 on CPU",
    )  # fmt: skip
    g.add_argument("--device", default=t.device, help="auto | cpu | cuda | mps")
    g.add_argument("--num-workers", type=int, default=t.num_workers, help="DataLoader workers")
    g.add_argument("--seed", type=int, default=t.seed, help="init and data order")
    g.add_argument("--log-every", type=int, default=t.log_every, help="steps between log lines")
    g.add_argument(
        "--eval-every", type=int, default=t.eval_every,
        help="steps between validations, 0 = once per epoch",
    )  # fmt: skip
    g.add_argument("--resume", help="checkpoint to continue from (e.g. .../last.pt)")
    g.add_argument(
        "--overfit", type=int,
        help="sanity check: train AND validate on the first N training images, no augmentation",
    )  # fmt: skip
    return p


def configs_from_args(args: argparse.Namespace) -> tuple[DataConfig, dict[str, Any], TrainConfig]:
    """CLI flags -> (DataConfig, model sizes, TrainConfig). num_classes comes from the data."""
    preset = DATASET_PRESETS[args.dataset]
    sizes = {
        k: getattr(args, k) if getattr(args, k) is not None else preset[k] for k in PRESET_FLAGS
    }
    data_config = DataConfig(
        dataset=args.dataset,
        data_dir=args.data_dir or f"data/{args.dataset}",
        classes=tuple(args.classes) if args.classes else None,
        max_per_class=args.max_per_class,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        image_size=sizes["image_size"],
        augment="none" if args.overfit else args.augment,
        shapes_per_class=args.shapes_per_class,
    )
    model_sizes = {
        **sizes,
        "mlp_ratio": args.mlp_ratio,
        "dropout": 0.0 if args.overfit else args.dropout,
        "patch_embed": args.patch_embed,
        "pool": args.pool,
    }
    train_config = TrainConfig(
        output_dir=args.output_dir,
        epochs=args.epochs,
        # an overfit set smaller than a batch would give zero full batches
        batch_size=min(args.batch_size, args.overfit) if args.overfit else args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        lr=args.lr,
        weight_decay=0.0 if args.overfit else args.weight_decay,
        warmup_steps=args.warmup_steps,
        label_smoothing=0.0 if args.overfit else args.label_smoothing,
        grad_clip=args.grad_clip,
        precision=args.precision,
        device=args.device,
        num_workers=args.num_workers,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        resume=args.resume,
        overfit=args.overfit,
    )
    return data_config, model_sizes, train_config


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_config, model_sizes, train_config = configs_from_args(args)
    info = setup_distributed(train_config.device)
    logging.basicConfig(
        level=logging.INFO if info.is_main else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        _build_and_train(data_config, model_sizes, train_config, info)
    finally:
        if info.enabled:
            dist.destroy_process_group()


def _build_and_train(
    data_config: DataConfig, model_sizes: dict[str, Any], train_config: TrainConfig, info: DistInfo
) -> TrainState:
    seed_everything(train_config.seed)
    with local_main_first(info):  # CIFAR-10 download happens once per machine
        datasets = build_datasets(data_config)
    train_set, val_set = datasets.train, datasets.val
    if train_config.overfit:
        # Same images for training and validation: the model should reach ~100%.
        # Balanced over classes: the first N items would all be of one class
        # (splits are grouped by class), and 100% on one class proves nothing.
        subset = balanced_subset(train_set.items, train_config.overfit)
        plain = build_transform(data_config, train=False)
        train_set = ImageDataset(subset, train_set.load, plain)
        val_set = ImageDataset(subset, train_set.load, plain)
        counts = Counter(datasets.class_names[label] for _, label in subset)
        logger.info("overfit on %d images: %s", len(subset), dict(sorted(counts.items())))
    logger.info(
        "data: %s | %d classes | train %d / val %d / test %d images",
        data_config.dataset, len(datasets.class_names), len(train_set), len(val_set),
        len(datasets.test),
    )  # fmt: skip

    model_config = ModelConfig(num_classes=len(datasets.class_names), **model_sizes)
    model = VisionTransformer(model_config)
    device = resolve_device(train_config.device)
    train_loader, val_loader = build_dataloaders(train_set, val_set, train_config, device)
    return train(model, train_loader, val_loader, data_config, train_config, datasets.class_names)


if __name__ == "__main__":
    main()
