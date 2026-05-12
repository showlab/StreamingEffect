# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import av
import numpy as np
import io
import re
import time
import torch
from typing import Optional, Union, Tuple
import random  # Import random for start time selection
from PIL import Image
from torchvision.transforms import Compose

import fastgen.utils.logging_utils as logger

# List of common video extensions PyAV might handle
VIDEO_EXTENSIONS = "mp4 ogv mjpeg avi mov h264 mpg webm wmv".split()
# List of common image extensions PIL might handle
IMAGE_EXTENSIONS = "jpg jpeg png bmp tif tiff webp gif".split()


def get_extension(key: str) -> str:
    return re.sub(r".*[.]", "", key).lower()


def decode_video_full(
    key: str,
    data: bytes,
    output_format: str = "torch",
    probesize: str = "1M",
    analyzeduration_us: int = 2_000_000,
    timeout_s: float = 3.0,
) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """
    Decode video bytes to video frames
    Args:
        key (str): The key (filename/extension) associated with the data.
        data (bytes): The video data as bytes.
        output_format (str): output format of the video frames
        probesize (str): probesize for ffmpeg
        analyzeduration_us (int): analyzeduration for ffmpeg
        timeout_s (float): timeout for ffmpeg
    Returns:
        np.ndarray | torch.Tensor | None: video frames array (T, H, W, 3) RGB or None on failure
    """
    if output_format not in ["numpy", "torch"]:
        raise ValueError("output_format must be either 'numpy' or 'torch'")

    if get_extension(key) not in VIDEO_EXTENSIONS:
        return None

    start = time.time()
    try:
        with av.open(
            io.BytesIO(data), mode="r", options={"probesize": probesize, "analyzeduration": str(analyzeduration_us)}
        ) as container:
            if not container.streams.video:
                return None
            video = container.streams.video[0]
            frames = []
            for frame in container.decode(video):
                if frame is None:
                    continue
                frames.append(frame.to_ndarray(format="rgb24"))
                if (time.time() - start) > timeout_s:
                    break
            if len(frames) == 0:
                logger.warning(f"Key '{key}': No frames collected after {timeout_s}s.")
                return None
            arr = np.asarray(frames)
            if output_format == "torch":
                arr = torch.from_numpy(arr)
            return arr
    except Exception as e:
        logger.error(f"Key '{key}': Error processing video with PyAV: {e}")
        return None


def decode_video_segment(
    key: str, data: bytes, num_frames: int, output_format: str = "torch"
) -> Optional[Union[np.ndarray, torch.Tensor]]:
    """Decodes a randomly selected segment of approximately num_frames from a video.

    Selects a random start time such that a clip of roughly num_frames can be
    extracted. If the video is shorter than num_frames, the entire video is
    decoded. Ensures that *at least* num_frames are returned if available after
    the random start time, truncating the result to exactly num_frames.

    Args:
        key (str): The key (filename/extension) associated with the data.
        data (bytes): The video data as bytes.
        num_frames (int): The target number of frames to decode.
        output_format (str): The desired output format ('numpy' or 'torch').
                             Defaults to 'torch'.

    Returns:
        np.ndarray | torch.Tensor | None: The decoded video frames (exactly num_frames
                                      if possible), or None if decoding fails.
        size: (num_frames, H, W, 3)
    """
    if output_format not in ["numpy", "torch"]:
        raise ValueError("output_format must be either 'numpy' or 'torch'")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")

    if get_extension(key) not in VIDEO_EXTENSIONS:
        return None

    frames = []

    try:
        # Use BytesIO to treat the byte data as a file
        with av.open(io.BytesIO(data), mode="r") as container:
            if not container.streams.video:
                logger.warning(f"Key '{key}': No video stream found.")
                return None
            video = container.streams.video[0]

            if video.time_base is None:
                logger.warning(f"Key '{key}': Video stream has no time_base. Cannot process.")
                return None

            # --- Calculate Durations and Random Start ---
            avg_fps = 0
            if video.average_rate:
                avg_fps = float(video.average_rate)
            elif video.guessed_rate:
                avg_fps = float(video.guessed_rate)

            total_duration_sec = 0
            if video.duration is not None:
                total_duration_sec = float(video.duration * video.time_base)

            if avg_fps > 0:
                # Add a "1 / avg_fps" buffer to account for FPS estimation inaccuracies
                estimated_needed_sec = (num_frames + 1) / avg_fps

                if total_duration_sec > estimated_needed_sec:
                    # Video is long enough, choose random start
                    max_start_sec = total_duration_sec - estimated_needed_sec
                    start_sec = random.uniform(0, max_start_sec)
                else:
                    # Video is shorter than needed, start from beginning
                    start_sec = 0
            else:
                # Cannot estimate FPS, default to reading from start
                start_sec = 0
                logger.warning(f"Key '{key}': Cannot determine FPS. Reading from start.")

            # --- Seek and Decode ---
            try:
                seek_pts = int(start_sec / video.time_base)
                container.seek(seek_pts, stream=video, backward=True, any_frame=False)
            except (av.error.ValueError, av.AVError) as seek_err:  # Catch specific AVError too
                logger.warning(
                    f"Key '{key}': Seek to {start_sec:.2f}s failed ({seek_err}). Attempting decode from beginning."
                )
                # Reset start_sec if seek fails, rely on time check loop
                start_sec = 0.0
                # Need to ensure the iterator starts from the beginning again if possible
                # Re-opening might be safest, but let's try continuing decode from start
                # container.seek(0, any_frame=False)? This might error too.
                # For simplicity, just continue decode loop and hope t<start_sec handles it
            except Exception as general_seek_err:
                logger.error(f"Key '{key}': Unexpected error during seek: {general_seek_err}")
                return None

            frame_count_after_start = 0
            for frame in container.decode(video):
                if frame.pts is None:
                    continue

                t = frame.time
                if t is None:
                    continue

                # Skip frames before our calculated start time
                if t < start_sec:
                    continue

                # We are at or after start_sec, collect frames
                try:
                    frames.append(frame.to_ndarray(format="rgb24"))
                    frame_count_after_start += 1
                    # If we have enough frames based on count, break early
                    # This helps if calculated_length_sec was overestimated
                    if frame_count_after_start >= num_frames:
                        break
                except Exception as frame_decode_err:
                    logger.warning(
                        f"Key '{key}': Error decoding frame at time {t:.2f}s: {frame_decode_err}. Skipping frame."
                    )

    except (av.AVError, IndexError, EOFError, ValueError) as e:
        logger.error(f"Key '{key}': Error processing video with PyAV: {e}")
        return None
    except Exception as e:
        logger.error(f"Key '{key}': Unexpected error processing video with PyAV: {e}")
        return None

    # --- Truncate and Finalize ---
    collected_count = len(frames)

    if collected_count == 0:
        logger.warning(f"Key '{key}': No frames collected after start time {start_sec:.2f}s.")
        # Return empty array/tensor
        empty_shape = (0, 1, 1, 3)  # Use a minimal valid shape
        return (
            np.zeros(empty_shape, dtype=np.uint8)
            if output_format == "numpy"
            else torch.zeros(empty_shape, dtype=torch.uint8)
        )

    # Truncate if we collected more than needed
    if collected_count > num_frames:
        frames = frames[:num_frames]

    # Log warning if we collected fewer than desired (and video wasn't inherently short)
    elif collected_count < num_frames:
        logger.warning(f"Key '{key}': Collected only {collected_count} frames, less than requested {num_frames}")
        pass

    try:
        result_array = np.asarray(frames)
        if output_format == "torch":
            return torch.from_numpy(result_array)
        else:
            return result_array
    except Exception as e:
        logger.error(f"Key '{key}': Error converting collected frames to array: {e}")
        return None


def decode_image(key: str, data: bytes, image_transform: Compose | None = None) -> Optional[Image.Image]:
    """
    Decodes .jpg files using custom logic.
    Returns None for other extensions to allow fall-through.
    """
    if get_extension(key) in IMAGE_EXTENSIONS:
        image = Image.open(io.BytesIO(data))
        if image_transform is not None:
            image = image_transform(image)
        return image
    return None


def decode_txt(key: str, data: bytes, txt_extensions: Tuple[str, ...] = ("txt",), strip: bool = True) -> Optional[str]:
    extension = get_extension(key)
    if extension in txt_extensions:
        txt = data.decode("utf-8")
        if strip:
            txt = txt.strip()
        return txt
    return None


def decode_npy_torch(key: str, data: bytes) -> Optional[torch.Tensor]:
    """Decode .npy bytes into a torch tensor."""
    if get_extension(key) == "npy":
        array = np.load(io.BytesIO(data))
        return torch.from_numpy(array)
    return None


def decode_npz_torch(key: str, data: bytes) -> Optional[dict]:
    """Decode .npz bytes into a dict of torch tensors."""
    if get_extension(key) == "npz":
        arrays = dict(np.load(io.BytesIO(data)))
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in arrays.items()}
    return None
