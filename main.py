import argparse

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.config import ALLOWED_DEFECTS
from src.io.utils import load_few_shot_dataset


def train(args):
    dataset = load_few_shot_dataset("pill", ALLOWED_DEFECTS)

    if args.distributed:
        # Initialize distributed mode
        dist.init_process_group(backend="nccl")  # "gloo" for CPU, "nccl" for GPU
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Create sampler
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )

        # Use the sampler in the DataLoader
        dataloader = DataLoader(
            dataset, batch_size=16, sampler=sampler, num_workers=4, pin_memory=True
        )
    else:
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=4,  # Use >0 for parallel data loading
            pin_memory=True,  # Optional: speed up transfer to GPU
        )

    # Iterate through batches
    for batch in dataloader:
        images = batch["image"]  # shape: (B, C, H, W)
        masks = batch["mask"]  # shape: (B, 1, H, W)
        captions = batch.get("caption")  # Optional


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a simulation or post-process an EpisodeStats pickle "
                    "and log the metrics to a CSV file."
    )
    p.add_argument(
        "--distributed",
        required=False,
        default=False,
        help="Whether to run distributed",
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
