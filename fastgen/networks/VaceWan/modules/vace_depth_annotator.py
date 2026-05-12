# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import torch
import torch.nn as nn
from typing import Optional

import fastgen.utils.logging_utils as logger

# Import VACE annotators
from fastgen.third_party.annotators.depth_anything_v2.dpt import DepthAnythingV2


class VACEDepthExtractor(nn.Module):
    """Depth extractor using VACE's Depth Anything V2 model.

    This integrates VACE's Depth Anything V2 model for
    on-the-fly depth extraction in FastGen.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize depth extractor with Depth Anything V2.

        Args:
            model_path: Path to model weights. If None, uses default path
            device: Device to run on. If None, auto-selects cuda/cpu
        """
        super().__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set default model path for Depth Anything V2
        if model_path is None:
            model_path = "FASTGEN_OUTPUT/.cache/annotators/depth_anything_v2_vitl.pth"

        # Initialize Depth Anything V2 model
        self.model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024]).to(self.device)

        # Load model weights if path exists
        if os.path.exists(model_path):
            net_load_info = self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=False)
            )
            logger.success(f"Loaded Depth Anything V2 model from {model_path}. Loading info: {net_load_info}")
        else:
            logger.warning(f"Depth Anything V2 model not found at {model_path}.")

        self.model.eval().requires_grad_(False)  # Freeze the depth model

    def process_frame_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Process a single frame tensor to extract depth.

        Args:
            image_tensor: Input tensor [C, H, W] in range [0, 1], RGB format

        Returns:
            Depth tensor [C, H, W] in range [0, 1], 3-channel grayscale
        """
        with torch.inference_mode():
            # Use the model's tensor-based inference
            depth = self.model.infer_tensor(image_tensor)  # Returns [H, W]

            # Normalize depth values
            depth_min = depth.min()
            depth_max = depth.max()

            # Handle case where all values are the same
            if depth_max - depth_min < 1e-8:
                depth_normalized = torch.ones_like(depth) * 0.5
            else:
                depth_normalized = (depth - depth_min) / (depth_max - depth_min)

            # Convert to 3-channel format by repeating
            depth_3ch = depth_normalized.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]

            return depth_3ch

    def extract_depth_from_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract depth maps from video frames.

        Args:
            frames: Video frames tensor [B, C, T, H, W] in range [0, 1], RGB format

        Returns:
            Depth maps tensor [B, C, T, H, W] in range [0, 1]
        """
        B, C, T, H, W = frames.shape

        # Process each frame in the batch
        all_depths = []

        for b in range(B):
            frame_depths = []

            for t in range(T):
                # Get single frame [C, H, W]
                frame = frames[b, :, t, :, :]

                # Process frame directly as tensor - no numpy conversion needed
                depth_tensor = self.process_frame_tensor(frame)

                frame_depths.append(depth_tensor)

            # Stack time dimension [C, T, H, W]
            batch_depth = torch.stack(frame_depths, dim=1)
            all_depths.append(batch_depth)

        # Stack batch dimension [B, C, T, H, W]
        depths = torch.stack(all_depths, dim=0)

        return depths

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract depth maps from input frames.

        Args:
            frames: Input tensor, either:
                - Video: [B, C, T, H, W] in RGB format
                - Images: [B, C, H, W] in RGB format
                Values should be in range [0, 1]

        Returns:
            Depth maps with same shape as input, in range [0, 1]
        """
        if frames.ndim == 5:
            # Video input
            return self.extract_depth_from_frames(frames)
        elif frames.ndim == 4:
            # Image batch - add time dimension, process, remove
            frames_5d = frames.unsqueeze(2)  # [B, C, 1, H, W]
            depths_5d = self.extract_depth_from_frames(frames_5d)
            return depths_5d.squeeze(2)  # [B, C, H, W]
        else:
            raise ValueError(f"Expected 4D or 5D input, got {frames.ndim}D")
