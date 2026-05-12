#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pre-encode V2V_background dataset into WebDataset TAR shards.

Source: /home/guian/data/V2V_background/Grounded-SAM-2/outputs/movie_new_citywalk/
Output: WebDataset TAR shards with pre-computed latents.

Per sample outputs in TAR:
  - target_latent.pth:      [48, F, H', W']   — full video (ground truth)
  - foreground_latent.pth:  [48, F, H', W']   — foreground condition
  - ref_latents.pth:        [48, 3, H', W']   — 3 ref images (zero-padded if <3)
  - ref_mask.pth:           [3]               — bool mask for valid refs
  - ref_frame_indices.json: list              — latent-space frame indices
  - txt_emb.pth:            [512, 4096]       — UMT5 text embedding
  - neg_txt_emb.pth:        [512, 4096]       — negative/empty prompt embedding

Usage:
    python scripts/precompute_v2v_bg_latents.py \
        --data_root /home/guian/data/V2V_background/Grounded-SAM-2/outputs/movie_new_citywalk \
        --output_dir /path/to/v2v_bg_latents \
        --model_id "Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
        --height 1056 --width 1920 --num_frames 97 \
        --samples_per_shard 100
"""

import argparse
import ast
import io
import json
import os
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torchvision.io as tvio
import torchvision.transforms.functional as TF
from tqdm import tqdm


def load_video_frames(video_path: str, num_frames: int, height: int, width: int) -> torch.Tensor:
    """Load and resize video frames.

    Returns: [C, T, H, W] tensor in [-1, 1] range.
    """
    vframes, _, _ = tvio.read_video(video_path, pts_unit="sec")
    # vframes: [T, H, W, C] uint8

    # Sample exactly num_frames uniformly
    total = vframes.shape[0]
    if total >= num_frames:
        indices = torch.linspace(0, total - 1, num_frames).long()
        vframes = vframes[indices]
    else:
        # Pad by repeating last frame
        pad = num_frames - total
        last_frame = vframes[-1:].expand(pad, -1, -1, -1)
        vframes = torch.cat([vframes, last_frame], dim=0)

    # [T, H, W, C] -> [C, T, H, W]
    vframes = vframes.permute(3, 0, 1, 2).float() / 255.0

    # Resize
    # [C, T, H, W] -> resize each frame
    C, T, H_orig, W_orig = vframes.shape
    if H_orig != height or W_orig != width:
        vframes = vframes.reshape(C * T, H_orig, W_orig).unsqueeze(0)
        vframes = torch.nn.functional.interpolate(
            vframes, size=(height, width), mode="bilinear", align_corners=False
        )
        vframes = vframes.squeeze(0).reshape(C, T, height, width)

    # Normalize to [-1, 1]
    vframes = vframes * 2.0 - 1.0
    return vframes


def load_ref_images(ref_dir: str, height: int, width: int) -> Tuple[Optional[torch.Tensor], List[int]]:
    """Load reference images from key_frames directory.

    Returns:
        ref_pixels: [C, N, H, W] tensor in [-1, 1] or None if no refs
        ref_frame_indices: list of original frame indices (from filename)
    """
    if not os.path.isdir(ref_dir):
        return None, []

    ref_files = sorted([f for f in os.listdir(ref_dir) if f.endswith(".png") or f.endswith(".jpg")])
    if len(ref_files) == 0:
        return None, []

    refs = []
    frame_indices = []
    for f in ref_files[:3]:  # Max 3 refs
        # Parse frame index from filename like "refined_final_010.png"
        parts = f.replace(".png", "").replace(".jpg", "").split("_")
        frame_idx = int(parts[-1])
        frame_indices.append(frame_idx)

        img_path = os.path.join(ref_dir, f)
        # Read image
        img = tvio.read_image(img_path).float() / 255.0  # [C, H, W]
        # Resize
        if img.shape[1] != height or img.shape[2] != width:
            img = TF.resize(img, [height, width], antialias=True)
        img = img * 2.0 - 1.0  # [-1, 1]
        refs.append(img)

    ref_pixels = torch.stack(refs, dim=1)  # [C, N, H, W]
    return ref_pixels, frame_indices


def encode_video(vae, pixels: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    """Encode video pixels to latent space with normalization.

    Args:
        vae: AutoencoderKLWan
        pixels: [C, T, H, W] in [-1, 1]
        latents_mean, latents_std: normalization constants from VAE config

    Returns:
        [z_dim, F, H', W'] normalized latent
    """
    with torch.no_grad():
        # VAE expects [B, C, T, H, W]
        x = pixels.unsqueeze(0).to(next(vae.parameters()).device, dtype=torch.float32)
        latent = vae.encode(x).latent_dist.sample()  # [1, z_dim, F, H', W']
        latent = (latent - latents_mean.to(latent.device)) / latents_std.to(latent.device)
        return latent.squeeze(0).cpu()


def encode_single_images(
    vae, pixels: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
) -> torch.Tensor:
    """Encode individual images (refs) without temporal compression.

    Args:
        pixels: [C, N, H, W] in [-1, 1]

    Returns:
        [z_dim, N, H', W']
    """
    device = next(vae.parameters()).device
    C, N, H, W = pixels.shape
    encoded_list = []
    with torch.no_grad():
        for i in range(N):
            x = pixels[:, i:i+1, :, :].unsqueeze(0).to(device, dtype=torch.float32)  # [1, C, 1, H, W]
            latent = vae.encode(x).latent_dist.sample()  # [1, z_dim, 1, H', W']
            latent = (latent - latents_mean.to(latent.device)) / latents_std.to(latent.device)
            encoded_list.append(latent.squeeze(0))  # [z_dim, 1, H', W']
    return torch.cat(encoded_list, dim=1).cpu()  # [z_dim, N, H', W']


def encode_text(text_encoder, tokenizer, prompt: str, device: torch.device) -> torch.Tensor:
    """Encode text prompt using UMT5.

    Returns: [512, 4096] text embedding
    """
    max_seq_len = 512
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_seq_len,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    ids = text_inputs.input_ids.to(device)
    mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        text_embeds = text_encoder(ids, mask).last_hidden_state  # [1, 512, 4096]

    seq_len = mask.gt(0).sum().long().item()
    # Zero out padding
    emb = text_embeds[0].cpu()  # [512, 4096]
    emb[seq_len:] = 0
    return emb


def tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def json_to_bytes(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="Pre-compute V2V_bg latents for WebDataset")
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root of V2V_background dataset (movie_new_citywalk)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for TAR shards")
    parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                        help="HuggingFace model ID for VAE and text encoder")
    parser.add_argument("--height", type=int, default=1056)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--num_frames", type=int, default=97,
                        help="Number of pixel frames to sample (97 -> 25 latent frames, 49 -> 13)")
    parser.add_argument("--samples_per_shard", type=int, default=100)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples to process (-1 for all)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting sample index (for resuming)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Total number of GPUs for parallel processing")
    parser.add_argument("--gpu_rank", type=int, default=0,
                        help="This GPU's rank (0-indexed)")
    parser.add_argument("--foreground_subdir", type=str, default="foreground_sam3",
                        help="Subdirectory for foreground videos")
    parser.add_argument("--target_subdir", type=str, default="full_videos",
                        help="Subdirectory for target (full) videos")
    parser.add_argument("--original_max_frame", type=int, default=120,
                        help="Max pixel-space frame index in original dataset (default: 120)")
    args = parser.parse_args()

    # Auto-set device from gpu_rank if using multi-GPU
    if args.num_gpus > 1 and args.device == "cuda:0":
        args.device = f"cuda:{args.gpu_rank}"

    device = torch.device(args.device)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("Loading VAE...")
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", cache_dir=hf_home, torch_dtype=torch.float32
    ).to(device).eval()
    vae.requires_grad_(False)

    latents_mean = torch.tensor(vae.config.latents_mean).float().view(1, vae.config.z_dim, 1, 1, 1)
    latents_std = torch.tensor(vae.config.latents_std).float().view(1, vae.config.z_dim, 1, 1, 1)

    print("Loading text encoder...")
    from transformers import AutoTokenizer, UMT5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", cache_dir=hf_home)
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder", cache_dir=hf_home, torch_dtype=torch.float32
    ).to(device).eval()
    text_encoder.requires_grad_(False)

    # Pre-compute negative prompt embedding (empty string)
    print("Computing negative prompt embedding...")
    neg_txt_emb = encode_text(text_encoder, tokenizer, "", device)

    # Load prompts
    prompts_file = os.path.join(args.data_root, "prompts.jsonl")
    prompts_dict = {}
    if os.path.exists(prompts_file):
        with open(prompts_file) as f:
            for line in f:
                entry = json.loads(line)
                line_idx = entry["line"]
                caption_data = entry.get("caption", "")
                if isinstance(caption_data, str):
                    try:
                        caption_obj = ast.literal_eval(caption_data)
                        caption = caption_obj.get("caption", caption_data) if isinstance(caption_obj, dict) else caption_data
                    except (ValueError, SyntaxError):
                        caption = caption_data
                else:
                    caption = str(caption_data)
                prompts_dict[line_idx] = caption

    # Find all video samples
    target_dir = os.path.join(args.data_root, args.target_subdir)
    foreground_dir = os.path.join(args.data_root, args.foreground_subdir)
    keyframes_dir = os.path.join(args.data_root, "key_frames")

    video_files = sorted([f for f in os.listdir(target_dir) if f.endswith(".mp4")])
    total_found = len(video_files)
    print(f"Found {total_found} video samples")

    if args.max_samples > 0:
        video_files = video_files[args.start_idx:args.start_idx + args.max_samples]
    else:
        video_files = video_files[args.start_idx:]

    # Split across GPUs if using multi-GPU
    if args.num_gpus > 1:
        chunk_size = (len(video_files) + args.num_gpus - 1) // args.num_gpus
        start = args.gpu_rank * chunk_size
        end = min(start + chunk_size, len(video_files))
        video_files = video_files[start:end]
        print(f"GPU {args.gpu_rank}/{args.num_gpus}: processing samples {start}-{end} ({len(video_files)} samples)")

    # Process and write TARs
    max_refs = 3
    original_max_frame = args.original_max_frame

    if args.num_gpus > 1:
        shard_idx = args.gpu_rank * 1000  # Give each GPU a non-overlapping shard range
    else:
        shard_idx = args.start_idx // args.samples_per_shard
    sample_count_in_shard = 0
    tar_writer = None
    processed = 0

    for vid_file in tqdm(video_files, desc="Processing samples"):
        vid_id = vid_file.replace(".mp4", "")
        line_idx = int(vid_id)

        # Open new shard if needed
        if sample_count_in_shard == 0 or sample_count_in_shard >= args.samples_per_shard:
            if tar_writer is not None:
                tar_writer.close()
            shard_path = os.path.join(args.output_dir, f"shard-{shard_idx:06d}.tar")
            tar_writer = tarfile.open(shard_path, "w")
            sample_count_in_shard = 0
            shard_idx += 1

        try:
            # Load target video
            target_path = os.path.join(target_dir, vid_file)
            target_pixels = load_video_frames(target_path, args.num_frames, args.height, args.width)

            # Load foreground video
            fg_path = os.path.join(foreground_dir, vid_file)
            if not os.path.exists(fg_path):
                print(f"  Skipping {vid_id}: no foreground video")
                continue
            fg_pixels = load_video_frames(fg_path, args.num_frames, args.height, args.width)

            # Load ref images
            ref_dir = os.path.join(keyframes_dir, vid_id)
            ref_pixels, raw_frame_indices = load_ref_images(ref_dir, args.height, args.width)

            # Encode target latent
            target_latent = encode_video(vae, target_pixels, latents_mean, latents_std)

            # Encode foreground latent
            fg_latent = encode_video(vae, fg_pixels, latents_mean, latents_std)

            # Encode ref latents (pad to 3)
            latent_max_frame = target_latent.shape[1] - 1  # 12 for 13 latent frames
            if ref_pixels is not None:
                ref_latent = encode_single_images(vae, ref_pixels, latents_mean, latents_std)
                num_refs = ref_latent.shape[1]
                ref_mask = torch.ones(max_refs, dtype=torch.bool)
                # Map frame indices to latent space
                ref_frame_indices = [
                    min(round(idx * latent_max_frame / original_max_frame), latent_max_frame)
                    for idx in raw_frame_indices
                ]
                # Pad to max_refs
                if num_refs < max_refs:
                    z_dim = ref_latent.shape[0]
                    h_lat, w_lat = ref_latent.shape[2], ref_latent.shape[3]
                    pad = torch.zeros(z_dim, max_refs - num_refs, h_lat, w_lat)
                    ref_latent = torch.cat([ref_latent, pad], dim=1)
                    ref_mask[num_refs:] = False
                    ref_frame_indices.extend([0] * (max_refs - num_refs))
            else:
                z_dim = target_latent.shape[0]
                h_lat, w_lat = target_latent.shape[2], target_latent.shape[3]
                ref_latent = torch.zeros(z_dim, max_refs, h_lat, w_lat)
                ref_mask = torch.zeros(max_refs, dtype=torch.bool)
                ref_frame_indices = [0] * max_refs

            # Encode text
            prompt = prompts_dict.get(line_idx, "")
            txt_emb = encode_text(text_encoder, tokenizer, prompt, device)

            # Write to TAR
            sample_key = f"{processed:08d}"

            def add_to_tar(name, data_bytes):
                info = tarfile.TarInfo(name=f"{sample_key}.{name}")
                info.size = len(data_bytes)
                tar_writer.addfile(info, io.BytesIO(data_bytes))

            add_to_tar("target_latent.pth", tensor_to_bytes(target_latent))
            add_to_tar("foreground_latent.pth", tensor_to_bytes(fg_latent))
            add_to_tar("ref_latents.pth", tensor_to_bytes(ref_latent))
            add_to_tar("ref_mask.pth", tensor_to_bytes(ref_mask))
            add_to_tar("ref_frame_indices.json", json_to_bytes(ref_frame_indices))
            add_to_tar("txt_emb.pth", tensor_to_bytes(txt_emb))
            add_to_tar("neg_txt_emb.pth", tensor_to_bytes(neg_txt_emb))

            sample_count_in_shard += 1
            processed += 1

        except Exception as e:
            print(f"  Error processing {vid_id}: {e}")
            continue

    if tar_writer is not None:
        tar_writer.close()

    print(f"\nDone! Processed {processed} samples into {shard_idx} shards")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
