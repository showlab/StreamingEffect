# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from torch.utils.data import DataLoader

from fastgen.datasets.class_cond_dataset import ImageFolderDataset
from fastgen.datasets.samplers import InfiniteSampler


class ImageLoader:
    def __init__(
        self,
        dataset_path: str,
        s3_path: str,
        batch_size: int,
        use_labels: bool = True,
        cache: bool = True,
        shuffle: bool = True,
        sampler_start_idx: int = 0,
        **kwargs,
    ):
        """
        ImageLoader for class conditional datasets
        Args:
            dataset_path (str): Path to the dataset
            s3_path (str): Path to the s3 bucket
            batch_size (int): Batch size
            use_labels (bool): Whether to use labels
            cache (bool): Whether to cache the dataset
            shuffle (bool): Whether to shuffle the dataset
            sampler_start_idx (int): Start index for the sampler
        """
        self.dataset = ImageFolderDataset(
            path=dataset_path, s3_path=s3_path, use_labels=use_labels, cache=cache, **kwargs
        )

        dataset_sampler = InfiniteSampler(dataset=self.dataset, shuffle=shuffle, start_idx=sampler_start_idx)
        data_loader_kwargs = dict(
            num_workers=1,  # don't change this, otherwise it will cause BadZipFile error
            pin_memory=True,
            prefetch_factor=2,
        )

        self.loader = DataLoader(
            dataset=self.dataset, sampler=dataset_sampler, batch_size=batch_size, **data_loader_kwargs
        )

    def __iter__(self):
        return iter(self.loader)
