#!/usr/bin/env python3
"""
Stage 2 (Self-Forcing distilled student) batch inference on the paired test-set.

Key differences from Stage 1 inference:
  - Sampler: 4-step Euler on x0-prediction (not 50-step UniPC).
  - t_list: [0.999, 0.937, 0.833, 0.624, 0.0] (training schedule).
  - CFG: off by default (distilled student absorbs teacher CFG).
    Pass --guidance_scale 5.0 to re-enable.

Each test sample is an MP4 whose first half = source video, second half = GT.
For an MP4 of length N we extract n_clips = min(N//2, N - N//2) // 97 clip pairs.

Outputs per clip:
  {stem}_clip{i}_pred.mp4     — predicted video
  {stem}_clip{i}_compare.mp4  — source | predicted | GT side-by-side

Supports multi-GPU inference via torchrun.

Usage (single GPU):
    python infer_stage2.py \
        --ckpt_path /path/to/stage2.ckpt \
        --output_dir ./outputs/stage2 \
        --testset /path/to/testset \
        --max_side 1088

Usage (multi-GPU):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 infer_stage2.py \
        --ckpt_path /path/to/stage2.ckpt \
        --output_dir ./outputs/stage2 \
        --testset /path/to/testset \
        --max_side 1088 \
        --guidance_scale 5.0
"""

import argparse
import gc
import os
import random
import sys
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import torch._dynamo
torch._dynamo.config.recompile_limit = 64
from pathlib import Path

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
NUM_FRAMES = 97
LATENT_FRAMES = 25
SPATIAL_SCALE = 16
NEG_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, "
    "paintings, images, static, overall gray, worst quality, low quality, JPEG compression "
    "residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background"
)

DEFAULT_T_LIST = [0.999, 0.937, 0.833, 0.624, 0.0]


def _float_or_none(s: str):
    if s is None:
        return None
    sl = s.strip().lower()
    if sl in ("none", "null", "off", "no", "false"):
        return None
    v = float(sl)
    return None if v == 0.0 else v


def _t_list_arg(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def target_resolution(h_orig: int, w_orig: int, max_side: int | None) -> tuple[int, int]:
    if max_side is not None:
        scale = max_side / max(h_orig, w_orig)
        h = int(h_orig * scale)
        w = int(w_orig * scale)
    else:
        h, w = h_orig, w_orig
    h = (h // 32) * 32
    w = (w // 32) * 32
    return h, w


def load_exact_frames(video_path: str, indices: list, height: int, width: int) -> torch.Tensor:
    import decord
    decord.bridge.set_bridge("torch")
    vr = decord.VideoReader(video_path)
    frames = vr.get_batch(indices).float() / 255.0 * 2.0 - 1.0
    frames = frames.permute(3, 0, 1, 2)
    C, T, H_orig, W_orig = frames.shape
    if H_orig != height or W_orig != width:
        scale = max(height / H_orig, width / W_orig)
        new_h = int(round(H_orig * scale))
        new_w = int(round(W_orig * scale))
        frames = F.interpolate(
            frames.reshape(C * T, 1, H_orig, W_orig),
            size=(new_h, new_w), mode="bilinear", align_corners=False,
        ).reshape(C, T, new_h, new_w)
        sh = (new_h - height) // 2
        sw = (new_w - width) // 2
        frames = frames[:, :, sh:sh + height, sw:sw + width]
    return frames


def load_png_as_ref(png_path: str, height: int, width: int) -> torch.Tensor:
    from PIL import Image
    img = Image.open(png_path).convert("RGB").resize((width, height), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0 * 2.0 - 1.0
    t = torch.from_numpy(arr).permute(2, 0, 1)
    return t.unsqueeze(1).unsqueeze(0)


def vae_encode(vae, pixels: torch.Tensor, latents_mean, latents_std) -> torch.Tensor:
    with torch.no_grad():
        z = vae.encode(pixels).latent_dist.sample()
        z = (z - latents_mean) / latents_std
    return z.to(torch.bfloat16)


def convert_fsdp_ckpt(ckpt_dir: str, iteration: int, output_pt: str, rank: int = 0):
    import torch.distributed as dist
    if rank == 0 and not os.path.exists(output_pt):
        script = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "scripts", "convert_fsdp_checkpoint.py")
        )
        cmd = [sys.executable, script,
               "--ckpt_dir", ckpt_dir,
               "--iteration", str(iteration),
               "--output", output_pt,
               "--key", "net"]
        print(f"[rank0] Converting FSDP ckpt iter {iteration} → {output_pt} ...")
        subprocess.run(cmd, check=True)
        print("[rank0] Conversion done.")
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def load_checkpoint(ckpt_path: str) -> dict:
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
        transformer_sd = {}
        for k, v in sd.items():
            for prefix in ("net.transformer.", "student_net.transformer."):
                if k.startswith(prefix):
                    transformer_sd[k[len(prefix):]] = v
                    break
        if transformer_sd:
            return transformer_sd
    if isinstance(ckpt, dict) and any(k.startswith("blocks.") for k in ckpt):
        return ckpt
    if isinstance(ckpt, dict) and any(k.startswith("transformer.") for k in ckpt):
        return {k[len("transformer."):]: v for k, v in ckpt.items() if k.startswith("transformer.")}
    raise ValueError(f"Unrecognized checkpoint format. Keys[:8]: {list(ckpt.keys())[:8]}")


def stage2_sample(net, *, noise, condition, neg_condition,
                  t_list_vals, guidance_scale, context_noise, sample_type="ode"):
    """Stage 2 inference: explicit t_list, x0-prediction Euler, optional CFG."""
    net.clear_caches()
    t_list = torch.tensor(t_list_vals, device=noise.device, dtype=net.noise_scheduler.t_precision)
    x = net.noise_scheduler.latents(noise=noise, t_init=t_list[0])
    return net._sample_euler(x, t_list, condition, neg_condition, guidance_scale, context_noise, sample_type)


def main():
    parser = argparse.ArgumentParser()
    ckpt_grp = parser.add_mutually_exclusive_group(required=True)
    ckpt_grp.add_argument("--ckpt_path", type=str, help="Pre-converted student .pt")
    ckpt_grp.add_argument("--ckpt_dir",  type=str, help="FSDP checkpoint directory")
    parser.add_argument("--ckpt_iter",      type=int,   default=3000)
    parser.add_argument("--output_dir",     type=str,   required=True)
    parser.add_argument("--testset",        type=str,   required=True,
                        help="Path to test set directory containing .mp4/.txt/(.png) files")
    parser.add_argument("--model_id",       type=str,   default=MODEL_ID)
    parser.add_argument("--max_side",       type=int,   default=1088,
                        help="Resize long side to this value (default: 1088)")
    parser.add_argument("--guidance_scale", type=_float_or_none, default=None,
                        help="CFG scale. Default: off. Pass float (e.g. 5.0) to enable.")
    parser.add_argument("--t_list",         type=_t_list_arg, default=DEFAULT_T_LIST,
                        help="Comma-separated t schedule ending in 0. "
                             "Default: [0.999,0.937,0.833,0.624,0.0]")
    parser.add_argument("--sample_type",    type=str,   default="ode", choices=["ode", "sde"])
    parser.add_argument("--ref_frame_idx",  type=int,   default=None,
                        help="Pin reference latent frame index. Default: random per clip.")
    parser.add_argument("--chunk_size",     type=int,   default=5)
    parser.add_argument("--context_noise",  type=float, default=0.0)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--skip_existing",  action="store_true", default=True)
    args = parser.parse_args()

    if len(args.t_list) < 2 or args.t_list[-1] != 0.0:
        raise ValueError(f"--t_list must have ≥2 values and end in 0.0, got {args.t_list}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank       = int(os.environ.get("RANK",       0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_dist = world_size > 1

    if is_dist:
        import torch.distributed as dist
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    os.makedirs(args.output_dir, exist_ok=True)

    if rank == 0:
        cfg_str = "OFF" if args.guidance_scale is None else f"ON (scale={args.guidance_scale})"
        print(f"\n=== Stage 2 inference ===")
        print(f"  t_list:  {args.t_list}  ({len(args.t_list) - 1} steps)")
        print(f"  CFG:     {cfg_str}")

    if args.ckpt_path:
        ckpt_path = args.ckpt_path
    else:
        ckpt_path = os.path.join(args.ckpt_dir, f"net_iter{args.ckpt_iter}.pt")
        convert_fsdp_ckpt(args.ckpt_dir, args.ckpt_iter, ckpt_path, rank=rank)

    from fastgen.networks.WanV2VBG.network_causal import CausalWanV2VBG
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ["HF_HOME"] = hf_home

    net = CausalWanV2VBG(
        model_id_or_local_path=args.model_id,
        disable_efficient_attn=False,
        disable_grad_ckpt=True,
        enable_logvar_linear=False,
        net_pred_type="flow",
        schedule_type="rf",
        load_pretrained=True,
        chunk_size=args.chunk_size,
        total_num_frames=LATENT_FRAMES,
        delete_cache_on_clear=True,
    )
    transformer_sd = load_checkpoint(ckpt_path)
    info = net.transformer.load_state_dict(transformer_sd, strict=False)
    if info.missing_keys and rank == 0:
        print(f"  WARNING: {len(info.missing_keys)} missing keys")
    net = net.to(device=device, dtype=torch.bfloat16).eval()

    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        args.model_id, subfolder="vae", cache_dir=hf_home, torch_dtype=torch.float32,
    ).to(device).eval()
    latents_mean = torch.tensor(vae.config.latents_mean).float().view(1, vae.config.z_dim, 1, 1, 1).to(device)
    latents_std  = torch.tensor(vae.config.latents_std ).float().view(1, vae.config.z_dim, 1, 1, 1).to(device)

    from transformers import AutoTokenizer, UMT5EncoderModel
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, subfolder="tokenizer", cache_dir=hf_home)
    text_encoder = UMT5EncoderModel.from_pretrained(
        args.model_id, subfolder="text_encoder", cache_dir=hf_home, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    def encode_text(text: str) -> torch.Tensor:
        tokens = tokenizer(text, max_length=512, padding="max_length",
                           truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = text_encoder(tokens.input_ids, attention_mask=tokens.attention_mask)[0]
        return emb * tokens.attention_mask.unsqueeze(-1).to(emb.dtype)

    testset = Path(args.testset)
    all_samples = sorted(p.stem for p in testset.glob("*.mp4"))
    my_samples = all_samples[rank::world_size]

    if rank == 0:
        print(f"\nFound {len(all_samples)} samples — {world_size} rank(s)\n")

    from diffusers.utils import export_to_video

    for idx, stem in enumerate(my_samples):
        mp4_path = str(testset / f"{stem}.mp4")
        txt_path = str(testset / f"{stem}.txt")

        import decord
        decord.bridge.set_bridge("torch")
        vr = decord.VideoReader(mp4_path)
        n_total = len(vr)
        h_orig, w_orig, _ = vr[0].shape
        half = n_total // 2
        n_clips = min(half, n_total - half) // NUM_FRAMES
        if n_clips == 0:
            continue

        H, W = target_resolution(h_orig, w_orig, args.max_side)
        print(f"[rank{rank}][{idx+1}/{len(my_samples)}] {stem}  {h_orig}x{w_orig} → {H}x{W}  clips={n_clips}")

        with open(txt_path, encoding="utf-8") as f:
            prompt = f.read().strip()

        text_emb = encode_text(prompt)
        neg_emb  = encode_text(NEG_PROMPT)

        png_path = str(testset / f"{stem}.png")
        png_ref_latent = None
        if os.path.exists(png_path):
            ref_pixels = load_png_as_ref(png_path, H, W).to(device)
            png_ref_latent = vae_encode(vae, ref_pixels.float(), latents_mean, latents_std)
            del ref_pixels

        for clip_i in range(n_clips):
            out_path = os.path.join(args.output_dir, f"{stem}_clip{clip_i}_pred.mp4")
            cmp_path = os.path.join(args.output_dir, f"{stem}_clip{clip_i}_compare.mp4")
            if args.skip_existing and os.path.exists(out_path):
                continue

            fg_indices = list(range(clip_i * NUM_FRAMES, (clip_i + 1) * NUM_FRAMES))
            gt_indices = [half + i for i in fg_indices]

            fg_pixels = load_exact_frames(mp4_path, fg_indices, H, W).unsqueeze(0).to(device)
            fg_latent = vae_encode(vae, fg_pixels.float(), latents_mean, latents_std)
            del fg_pixels

            if png_ref_latent is not None:
                ref_latent = png_ref_latent
            else:
                ref_pixels = load_exact_frames(mp4_path, [gt_indices[-1]], H, W).unsqueeze(0).to(device)
                ref_latent = vae_encode(vae, ref_pixels.float(), latents_mean, latents_std)
                del ref_pixels

            ref_idx = args.ref_frame_idx if args.ref_frame_idx is not None else random.randrange(LATENT_FRAMES)
            ref_mask = torch.ones(1, 1, device=device)

            condition = {
                "text_embeds":       text_emb,
                "foreground_latent": fg_latent,
                "ref_latents":       ref_latent,
                "ref_mask":          ref_mask,
                "ref_frame_indices": [ref_idx],
            }
            neg_condition = {
                "text_embeds":       neg_emb,
                "foreground_latent": fg_latent,
                "ref_latents":       ref_latent,
                "ref_mask":          ref_mask,
                "ref_frame_indices": [ref_idx],
            }

            noise = torch.randn(
                1, vae.config.z_dim, LATENT_FRAMES,
                H // SPATIAL_SCALE, W // SPATIAL_SCALE,
                device=device, dtype=torch.bfloat16,
            )
            with torch.no_grad():
                latents = stage2_sample(
                    net,
                    noise=noise,
                    condition=condition,
                    neg_condition=neg_condition,
                    t_list_vals=args.t_list,
                    guidance_scale=args.guidance_scale,
                    context_noise=args.context_noise,
                    sample_type=args.sample_type,
                )

            latents_cpu = latents.cpu()
            net.clear_caches()
            del noise, latents, condition, neg_condition, fg_latent, ref_mask
            if png_ref_latent is None:
                del ref_latent
            gc.collect()
            torch.cuda.empty_cache()

            latents_dev = latents_cpu.to(device=device, dtype=vae.dtype) * latents_std + latents_mean
            with torch.no_grad():
                pred_video = vae.decode(latents_dev, return_dict=False)[0]
            del latents_dev, latents_cpu

            pred_np = pred_video.squeeze(0).permute(1, 2, 3, 0).float().cpu().numpy()
            pred_np = (pred_np.clip(-1, 1) * 0.5 + 0.5).clip(0, 1)
            del pred_video
            gc.collect()
            torch.cuda.empty_cache()

            export_to_video(pred_np, output_video_path=out_path, fps=24)

            fg_np = load_exact_frames(mp4_path, fg_indices, H, W)
            fg_np = (fg_np.permute(1, 2, 3, 0).float().numpy().clip(-1, 1) * 0.5 + 0.5).clip(0, 1)
            gt_np = load_exact_frames(mp4_path, gt_indices, H, W)
            gt_np = (gt_np.permute(1, 2, 3, 0).float().numpy().clip(-1, 1) * 0.5 + 0.5).clip(0, 1)
            export_to_video(np.concatenate([fg_np, pred_np, gt_np], axis=2),
                            output_video_path=cmp_path, fps=24)
            del fg_np, gt_np, pred_np
            gc.collect()

        del text_emb, neg_emb
        if png_ref_latent is not None:
            del png_ref_latent
        gc.collect()
        torch.cuda.empty_cache()

    if is_dist:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"\nDone. Results in: {args.output_dir}")


if __name__ == "__main__":
    main()
