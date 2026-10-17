"""Multi-process (DDP) training on CPU.

Starts 2 processes with the gloo backend, exactly like torchrun would on 2 GPUs,
and checks they stay in sync: same step count and identical final weights.
"""

import os
import socket
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from conftest import folder_data_config, make_image_folder, tiny_model_config, tiny_train_config

from tiny_vit.data import build_datasets
from tiny_vit.model import VisionTransformer
from tiny_vit.train import LAST_CHECKPOINT, build_dataloaders, get_dist_info, train

WORLD_SIZE = 2


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def worker(rank: int, port: int, tmp_path: Path) -> None:
    """What torchrun would run in each process."""
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(WORLD_SIZE),
    )  # fmt: skip
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    try:
        assert get_dist_info().world_size == WORLD_SIZE
        torch.manual_seed(rank)  # different init per rank: DDP must broadcast rank 0's
        model = VisionTransformer(tiny_model_config())
        data_config = folder_data_config(tmp_path / "images")
        datasets = build_datasets(data_config)
        config = tiny_train_config(tmp_path / "run", epochs=3, batch_size=2, grad_accum_steps=2)
        train_loader, val_loader = build_dataloaders(
            datasets.train, datasets.val, config, torch.device("cpu")
        )
        # 24 train images / 2 processes / batch 2 = 6 batches per process.
        assert len(train_loader) == 6
        state = train(model, train_loader, val_loader, data_config, config, datasets.class_names)
        torch.save({"step": state.step, "model": model.state_dict()}, tmp_path / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_two_process_training_stays_in_sync(tmp_path: Path) -> None:
    make_image_folder(tmp_path / "images")
    mp.spawn(worker, args=(free_port(), tmp_path), nprocs=WORLD_SIZE, join=True)

    rank0 = torch.load(tmp_path / "rank0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank1.pt", weights_only=True)
    # 6 batches / 2 accumulation = 3 optimizer steps per epoch, 3 epochs.
    assert rank0["step"] == rank1["step"] == 9
    for name, weight in rank0["model"].items():
        torch.testing.assert_close(weight, rank1["model"][name], msg=f"{name} differs")
    assert (tmp_path / "run" / LAST_CHECKPOINT).exists()  # written by rank 0 only
