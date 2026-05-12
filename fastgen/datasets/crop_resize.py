# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from PIL import Image

import torch
import torchvision.transforms.functional as transforms_F

import fastgen.utils.logging_utils as logger


def obtain_image_size(data: Any) -> tuple[int, int]:
    """Function for obtaining the image size from the data.

    Args:
        data: Input data
    Returns:
        width (int): Width of the input image
        height (int): Height of the input image
    """

    if isinstance(data, Image.Image):
        width, height = data.size
    elif isinstance(data, torch.Tensor):
        height, width = data.size()[-2:]
    else:
        raise ValueError("data to random crop should be PIL Image or tensor")

    return width, height


# ------------------------------------------------------------------------
# crop


def center_crop(data: Any, img_size: tuple[int, int], return_cropping_params: bool = False) -> Any:
    """Performs center crop."""
    img_w, img_h = img_size

    orig_w, orig_h = obtain_image_size(data)
    data = transforms_F.center_crop(data, [img_h, img_w])

    crop_x0 = (orig_w - img_w) // 2
    crop_y0 = (orig_h - img_h) // 2
    cropping_params = {
        "resize_w": orig_w,
        "resize_h": orig_h,
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_w": img_w,
        "crop_h": img_h,
    }

    if return_cropping_params:
        return data, cropping_params
    return data


def random_crop(data: Any, img_size: tuple[int, int], return_cropping_params: bool = False) -> Any:
    """Performs random crop."""
    img_w, img_h = img_size

    orig_w, orig_h = obtain_image_size(data)
    # Obtaining random crop coords
    try:
        crop_x0 = int(torch.randint(0, orig_w - img_w + 1, size=(1,)).item())
        crop_y0 = int(torch.randint(0, orig_h - img_h + 1, size=(1,)).item())
    except Exception:
        logger.warning(
            f"Random crop failed. Performing center crop, original_size(wxh): {orig_w}x{orig_h}, random_size(wxh): {img_w}x{img_h}"
        )
        crop_x0 = (orig_w - img_w) // 2
        crop_y0 = (orig_h - img_h) // 2

    data = transforms_F.crop(data, crop_y0, crop_x0, img_h, img_w)
    cropping_params = {
        "resize_w": orig_w,
        "resize_h": orig_h,
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_w": img_w,
        "crop_h": img_h,
    }

    if return_cropping_params:
        return data, cropping_params
    return data


# ------------------------------------------------------------------------
# resize


def resize_small_side_aspect_preserving(data: Any, img_size: tuple[int, int]) -> Any:
    """
    Performs aspect-ratio preserving resizing.
    Image is resized to the dimension which has the larger ratio of (size / target_size).
    """
    img_w, img_h = img_size

    orig_w, orig_h = obtain_image_size(data)
    scaling_ratio = max((img_w / orig_w), (img_h / orig_h))
    target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

    assert (
        target_size[0] >= img_h and target_size[1] >= img_w
    ), f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"

    data = transforms_F.resize(
        data,
        size=target_size,  # type: ignore
        interpolation=transforms_F.InterpolationMode.BICUBIC,
        antialias=True,
    )
    return data


def resize_large_side_aspect_preserving(data: Any, img_size: tuple[int, int]) -> Any:
    """
    Performs aspect-ratio preserving resizing.
    Image is resized to the dimension which has the larger ratio of (size / target_size).
    """
    img_w, img_h = img_size

    orig_w, orig_h = obtain_image_size(data)
    scaling_ratio = min((img_w / orig_w), (img_h / orig_h))
    target_size = (int(scaling_ratio * orig_h + 0.5), int(scaling_ratio * orig_w + 0.5))

    assert (
        target_size[0] <= img_h and target_size[1] <= img_w
    ), f"Resize error. orig {(orig_w, orig_h)} desire {img_size} compute {target_size}"

    data = transforms_F.resize(
        data,
        size=target_size,  # type: ignore
        interpolation=transforms_F.InterpolationMode.BICUBIC,
        antialias=True,
    )
    return data
