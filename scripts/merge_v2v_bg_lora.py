#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Merge V2V_background rank-80 LoRA weights into the base Wan2.2-TI2V-5B model.

Usage:
    python scripts/merge_v2v_bg_lora.py \
        --base_model_id "/path/to/Wan2.2-TI2V-5B-Diffusers" \
        --lora_ckpt_path /path/to/stage0.ckpt \
        --output_path /path/to/merged_transformer.pt

The output is a merged state_dict that can be loaded directly into
WanTransformer3DModel (diffusers) or used as pretrained weights for CausalWanV2VBG.
"""

import argparse
import os
import torch
from diffusers.models import WanTransformer3DModel


def extract_lora_from_lightning(ckpt_path: str) -> dict:
    """Extract LoRA state_dict from a PyTorch Lightning checkpoint.

    V2V_background checkpoint structure:
    {
        "state_dict": {
            "lora": {
                "blocks.0.attn1.to_q.lora_A.default.weight": ...,
                "blocks.0.attn1.to_q.lora_B.default.weight": ...,
                ...
            }
        }
    }

    Also handles the older format where LoRA keys are directly in state_dict
    with "transformer." prefix.
    """
    print(f"Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    # Check for nested "lora" dict (V2V_background format)
    if "lora" in state_dict and isinstance(state_dict["lora"], dict):
        print(f"  Found nested 'lora' dict with {len(state_dict['lora'])} keys")
        return state_dict["lora"]

    # Fall back to extracting LoRA keys from flat state_dict
    lora_sd = {}
    for k, v in state_dict.items():
        if "lora_A" in k or "lora_B" in k:
            # Remove "transformer." prefix if present
            new_key = k[len("transformer."):] if k.startswith("transformer.") else k
            lora_sd[new_key] = v

    print(f"  Extracted {len(lora_sd)} LoRA keys from flat state_dict")
    return lora_sd


def merge_lora_into_base(base_sd: dict, lora_sd: dict, lora_alpha: int = 80, lora_rank: int = 80) -> dict:
    """Merge LoRA weights into base model weights.

    LoRA modifies weight as: W' = W + (alpha/rank) * B @ A
    where A is lora_A and B is lora_B.
    """
    merged_sd = dict(base_sd)
    scale = lora_alpha / lora_rank

    # Find all LoRA A matrices
    lora_a_keys = [k for k in lora_sd if "lora_A" in k]
    merged_count = 0

    for a_key in lora_a_keys:
        # Construct the corresponding B key and base weight key
        b_key = a_key.replace("lora_A", "lora_B")
        # e.g., "blocks.0.attn1.to_q.lora_A.default.weight" -> "blocks.0.attn1.to_q.weight"
        base_key = a_key.split(".lora_A")[0] + ".weight"

        if b_key not in lora_sd:
            print(f"  Warning: Missing lora_B for {a_key}, skipping")
            continue

        if base_key not in merged_sd:
            # Try without .weight suffix variations
            base_key_alt = a_key.split(".lora_A")[0] + ".base_layer.weight"
            if base_key_alt in lora_sd:
                # The base weight might be in the LoRA checkpoint itself
                merged_sd[base_key.replace(".base_layer", "")] = lora_sd[base_key_alt]
                base_key = base_key.replace(".base_layer", "")
            else:
                print(f"  Warning: Base weight {base_key} not found, skipping")
                continue

        A = lora_sd[a_key].float()  # [rank, in_features]
        B = lora_sd[b_key].float()  # [out_features, rank]
        W = merged_sd[base_key].float()

        # Merge: W' = W + scale * B @ A
        delta = scale * (B @ A)
        merged_sd[base_key] = (W + delta).to(merged_sd[base_key].dtype)
        merged_count += 1

    print(f"  Merged {merged_count} LoRA weight pairs (scale={scale})")
    return merged_sd


def main():
    parser = argparse.ArgumentParser(description="Merge V2V_background LoRA into base Wan2.2")
    parser.add_argument("--base_model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                        help="HuggingFace model ID or local path for base Wan2.2")
    parser.add_argument("--lora_ckpt_path", type=str, required=True,
                        help="Path to V2V_background Lightning checkpoint (.ckpt)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for merged transformer state_dict (.pt)")
    parser.add_argument("--lora_rank", type=int, default=80,
                        help="LoRA rank (default: 80)")
    parser.add_argument("--lora_alpha", type=int, default=80,
                        help="LoRA alpha (default: 80, same as rank)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify merge by loading the merged weights")
    args = parser.parse_args()

    # Step 1: Load base model
    print(f"Loading base model from {args.base_model_id}")
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    base_model = WanTransformer3DModel.from_pretrained(
        args.base_model_id,
        subfolder="transformer",
        cache_dir=hf_home,
        torch_dtype=torch.float32,
    )
    base_sd = base_model.state_dict()
    print(f"  Base model has {len(base_sd)} parameters")
    del base_model  # Free memory

    # Step 2: Load LoRA weights from Lightning checkpoint
    lora_sd = extract_lora_from_lightning(args.lora_ckpt_path)
    lora_a_keys = [k for k in lora_sd if "lora_A" in k]
    lora_b_keys = [k for k in lora_sd if "lora_B" in k]
    print(f"  Found {len(lora_a_keys)} LoRA_A + {len(lora_b_keys)} LoRA_B keys")

    # Step 3: Merge LoRA into base
    print("Merging LoRA weights into base model...")
    merged_sd = merge_lora_into_base(
        base_sd, lora_sd,
        lora_alpha=args.lora_alpha,
        lora_rank=args.lora_rank,
    )

    # Step 4: Save merged state_dict
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    print(f"Saving merged state_dict to {args.output_path}")
    torch.save(merged_sd, args.output_path)
    print(f"  Saved {len(merged_sd)} parameters")

    # Step 5: Optional verification
    if args.verify:
        print("Verifying merged weights...")
        verify_model = WanTransformer3DModel.from_pretrained(
            args.base_model_id,
            subfolder="transformer",
            cache_dir=hf_home,
            torch_dtype=torch.float32,
        )
        load_info = verify_model.load_state_dict(merged_sd, strict=False)
        print(f"  Missing keys: {load_info.missing_keys}")
        print(f"  Unexpected keys: {load_info.unexpected_keys}")
        if len(load_info.missing_keys) == 0 and len(load_info.unexpected_keys) == 0:
            print("  Verification PASSED: All keys match!")
        else:
            print("  Verification WARNING: Some keys don't match")
        del verify_model

    print("Done!")


if __name__ == "__main__":
    main()
