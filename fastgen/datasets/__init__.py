# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import webdataset as wds

import torch
from torch.utils.data import default_collate


def _nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
    else:
        yield from src


def _collate_fn(batch):
    # Filter out None samples and collate
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return {}
    return default_collate(valid_batch)


def batch_wds_dataset(dataset: wds.WebDataset, batch_size: int, partial: bool = False) -> wds.WebDataset:
    """Batch a webdataset.

    Args:
        dataset (wds.WebDataset): Input webdataset.
        batch_size (int): Batchsize of the loader.
        partial (bool, optional): Whether to return partial batches. Defaults to False.

    Returns:
        wds.WebDataset: Batched webdataset.
    """
    return dataset.batched(batch_size, partial=partial, collation_fn=_collate_fn)


# ---------------------------------- Webdataset for training ----------------------------------


def wds_to_dataloader(
    dataset: wds.WebDataset,
    batch_size: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
    partial: bool = False,
) -> wds.WebLoader:
    """Wrap a webdataset with the DataLoader interface.

    Args:
        dataset (wds.WebDataset): Input webdataset.
        batch_size (int): Batchsize of the loader. If None, assumes that the dataset is already batched.
        num_workers (int): Number of workers for parallel dataprocessing. Defaults to 0.
        pin_memory (bool, optional): Whether to pin memory in loader. Defaults to True.
        partial (bool, optional): Whether to return partial batches. Defaults to False.

    Returns:
        wds.WebLoader: WebLoader wrapping the input webdataset.
    """
    if batch_size is not None:
        dataset = batch_wds_dataset(dataset, batch_size, partial)

    data_loader = wds.WebLoader(
        dataset,
        batch_size=None,
        collate_fn=None,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return data_loader
