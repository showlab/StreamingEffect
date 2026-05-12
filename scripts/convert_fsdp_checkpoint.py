#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Convert FSDP distributed checkpoint to a regular PyTorch state_dict.

FSDP training saves checkpoints as:
  checkpoints/0001000.pth            (metadata: scheduler, iteration, ...)
  checkpoints/0001000.net_model/     (sharded model weights)
  checkpoints/0001000.net_optim/     (sharded optimizer state)

This script consolidates the sharded weights into a single .pt file
that can be used with:
  - pretrained_model_path (for Causal CD / Self-Forcing teacher/student init)
  - inference_v2v_bg_causal.py --ckpt_path

Usage:
    # Convert net weights from iteration 1000
    python scripts/convert_fsdp_checkpoint.py \
        --ckpt_dir /path/to/checkpoints \
        --iteration 1000

    # Or specify the model directory directly
    python scripts/convert_fsdp_checkpoint.py \
        --model_dir /path/to/checkpoints/0001000.net_model

    # Custom output path
    python scripts/convert_fsdp_checkpoint.py \
        --ckpt_dir /path/to/checkpoints \
        --iteration 1000 \
        --output /path/to/net.pt
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to regular state_dict")
    parser.add_argument("--ckpt_dir", type=str, default=None,
                        help="Checkpoint directory containing XXXXXXX.pth and XXXXXXX.net_model/")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Training iteration to convert (e.g., 1000)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Direct path to .net_model/ directory (alternative to --ckpt_dir + --iteration)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .pt file path (default: <ckpt_dir>/net_iter<iteration>.pt)")
    parser.add_argument("--key", type=str, default="net",
                        help="Model key to extract (default: 'net'). Use 'ema' for EMA weights.")
    args = parser.parse_args()

    # Resolve model directory
    if args.model_dir is not None:
        model_dir = args.model_dir
        # Infer output path
        if args.output is None:
            parent = os.path.dirname(model_dir)
            basename = os.path.basename(model_dir).replace(f".{args.key}_model", "")
            args.output = os.path.join(parent, f"{args.key}_iter{basename}.pt")
    elif args.ckpt_dir is not None and args.iteration is not None:
        prefix = f"{args.iteration:07d}"
        model_dir = os.path.join(args.ckpt_dir, f"{prefix}.{args.key}_model")
        if args.output is None:
            args.output = os.path.join(args.ckpt_dir, f"{args.key}_iter{args.iteration}.pt")
    else:
        parser.error("Provide either --model_dir or both --ckpt_dir and --iteration")

    if not os.path.isdir(model_dir):
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)

    print(f"Converting FSDP checkpoint:")
    print(f"  Source: {model_dir}")
    print(f"  Output: {args.output}")

    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
    dcp_to_torch_save(model_dir, args.output)

    # Verify
    import torch
    sd = torch.load(args.output, map_location="cpu", weights_only=True)
    print(f"  Keys: {len(sd)}")
    prefixes = sorted(set(k.split('.')[0] for k in sd.keys()))
    print(f"  Top-level modules: {prefixes}")
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"  File size: {size_mb:.0f} MB")
    print("Done.")


if __name__ == "__main__":
    main()
