"""推理脚本：testset 推理版

在 wan2_inference_plus.py 基础上修改：
1. 输出保留输入目录结构：output_root/exp_name/infer_samples/<dataset_basename>/<rel_subdir>/<base>_clip{idx}.mp4
2. 支持断点续推：启动前过滤掉输出已存在的样本
3. 文件名一律加 _clip{idx} 后缀（无论该视频是否被切多段）
"""
import os
import argparse
import copy
import warnings
import numpy as np
import torch
from omegaconf import OmegaConf

import pytorch_lightning as L
from pytorch_lightning.utilities import rank_zero_only

from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import FlowMatchEulerDiscreteScheduler

from models.wan2.transformer_wan import WanTransformer3DModel
from models.wan2.custom_pipeline import CustomWanPipeline as WanPipeline
from models.wan2.attn_process import ConditionAttnProcessor2_0

from tools.my_schedule import FlowMatchScheduler, MyFlowMatchEulerDiscreteScheduler
from datasets.custom_dataset import CustomDataset, BucketSampler


@rank_zero_only
def silence_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _resolve_save_path(video_path: str, dataset_roots, save_root: str, clip_idx: int) -> str:
    """根据 video_path 与匹配到的 dataset_root 计算保留目录结构的输出路径。"""
    vp_real = os.path.realpath(video_path)
    matched_root = None
    for r in dataset_roots:
        if not r:
            continue
        r_real = os.path.realpath(r)
        if vp_real == r_real:
            matched_root = r_real
            break
        if vp_real.startswith(r_real.rstrip(os.sep) + os.sep):
            matched_root = r_real
            break

    if matched_root is None:
        sub_root = "misc"
        rel_dir = ""
        base = os.path.splitext(os.path.basename(video_path))[0]
    else:
        sub_root = os.path.basename(matched_root.rstrip(os.sep))
        rel = os.path.relpath(vp_real, matched_root)
        rel_dir = os.path.dirname(rel)
        base = os.path.splitext(os.path.basename(rel))[0]

    fname = f"{base}_clip{int(clip_idx)}.mp4"
    return os.path.join(save_root, sub_root, rel_dir, fname)


class InteractionVideoSystemInfer(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.hparams = opt
        self.is_configured = False

    def _align_frames(self, meta, gen, gt, is_one2three: bool):
        f_meta, f_gen, f_gt = meta.shape[0], gen.shape[0], gt.shape[0]
        if is_one2three and f_meta == f_gt and f_gen == f_meta - 1:
            meta = meta[1:]
            gt = gt[1:]
            return meta, gen, gt
        min_f = min(f_meta, f_gen, f_gt)
        return meta[:min_f], gen[:min_f], gt[:min_f]

    def configure_model(self):
        if self.is_configured:
            return
        self.is_configured = True

        model_id = self.hparams.model_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.float32
        )

        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32
        )

        if self.hparams.use_DiffSynth:
            self.train_scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
            self.train_scheduler.set_timesteps(1000, training=True)
        else:
            self.train_scheduler = MyFlowMatchEulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler"
            )
        base_sampler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.sample_scheduler = UniPCMultistepScheduler.from_config(
            base_sampler.config, flow_shift=5
        )

        self.transformer = WanTransformer3DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.float32
        )

        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.transformer.requires_grad_(False)

        if getattr(self.hparams.training, "gradient_checkpointing", False):
            self.transformer.gradient_checkpointing = True
            self.transformer.enable_gradient_checkpointing()

        self.register_buffer(
            'latents_mean',
            torch.tensor(self.vae.config.latents_mean).float().view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )
        self.register_buffer(
            'latents_std',
            torch.tensor(self.vae.config.latents_std).float().view(1, self.vae.config.z_dim, 1, 1, 1),
            persistent=False
        )

        self.vae_config = self.vae.config
        self.model_config = self.transformer.module.config if hasattr(self.transformer, "module") else self.transformer.config

        self.using_lora = bool(self.hparams.use_lora)
        if self.using_lora:
            from peft import LoraConfig
            transformer_lora_config = LoraConfig(
                r=96, lora_alpha=96, init_lora_weights=True,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.transformer.add_adapter(transformer_lora_config)

        for blk in self.transformer.blocks:
            blk.attn1.set_processor(ConditionAttnProcessor2_0())

        self.transformer.patch_embedding_extra = copy.deepcopy(self.transformer.patch_embedding).requires_grad_(True)

    @torch.no_grad()
    def encode_prompt(self, prompt_list, device):
        max_sequence_length = 512
        text_inputs = self.tokenizer(
            prompt_list,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids, mask = text_inputs.input_ids.to(device), text_inputs.attention_mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        text_embeds = self.text_encoder(ids, mask).last_hidden_state
        text_embeds = [u[:v] for u, v in zip(text_embeds, seq_lens)]
        text_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in text_embeds], dim=0
        )
        return text_embeds

    def _load_lora_from_ckpt(self, ckpt_path, device):
        if ckpt_path in (None, "", "None", "null"):
            print("[Infer] No ckpt_path provided. Skip loading LoRA.")
            return

        print(f"[Infer] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        if "state_dict" not in ckpt:
            print("[Infer] checkpoint has no 'state_dict' key; skip.")
            return

        sd_all = ckpt["state_dict"]

        if "transformer_processor" in sd_all:
            sd = sd_all["transformer_processor"]
            cur = self.transformer.state_dict()
            filtered = {k: v for k, v in sd.items() if (k in cur and cur[k].shape == v.shape)}
            skipped = [k for k in sd.keys() if k not in filtered]
            print(f"[Infer][LoRA] Load {len(filtered)}/{len(sd)} keys. Skipped {len(skipped)} mismatched keys.")
            self.transformer.load_state_dict(filtered, strict=False)
        else:
            print("[Infer] 'transformer_processor' not found in ckpt.state_dict; skip LoRA.")

        if "patch_embedding_extra" in sd_all:
            sd2 = sd_all["patch_embedding_extra"]
            cur2 = self.transformer.state_dict()
            filtered2 = {k: v for k, v in sd2.items() if (k in cur2 and cur2[k].shape == v.shape)}
            skipped2 = [k for k in sd2.keys() if k not in filtered2]
            print(f"[Infer][patch_embedding_extra] Load {len(filtered2)}/{len(sd2)} keys. Skipped {len(skipped2)} mismatched keys.")
            self.transformer.load_state_dict(filtered2, strict=False)
        else:
            print("[Infer] 'patch_embedding_extra' not found in ckpt.state_dict; skip.")

    @torch.no_grad()
    def run_infer(self):
        if not torch.distributed.is_initialized():
            if torch.distributed.is_available() and int(os.environ.get("WORLD_SIZE", 1)) > 1:
                torch.distributed.init_process_group(backend="nccl")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)

        self.configure_model()
        self.to(device)

        dataset_cfg = self.hparams.dataset
        video_root = dataset_cfg.get("video_root", "")
        video_root2 = dataset_cfg.get("video_root2", "")
        first_root = dataset_cfg.get("first_root", "")
        dataset_roots = list(dataset_cfg.get("dataset_roots", []) or [])
        cache_index_path = dataset_cfg.get("cache_index_path", None)
        use_bucket = bool(dataset_cfg.get("use_bucket_training", False))
        bucket_align = int(dataset_cfg.get("bucket_align", 32))
        max_long_side = int(dataset_cfg.get("max_long_side", 0))

        ds = CustomDataset(
            video_root=video_root,
            video_root2=video_root2,
            first_root=first_root,
            dataset_roots=dataset_roots,
            cache_index_path=cache_index_path,
            height=dataset_cfg.height,
            width=dataset_cfg.width,
            sample_n_frames=dataset_cfg.sample_n_frames,
            is_one2three=dataset_cfg.is_one2three,
            use_bucket_training=use_bucket,
            bucket_align=bucket_align,
            max_long_side=max_long_side,
            index_num_workers=int(dataset_cfg.get("index_num_workers", 8)),
            skip_first_clip=bool(dataset_cfg.get("skip_first_clip", False)),
            use_tail_as_ref=bool(dataset_cfg.get("use_tail_as_ref", False)),
            ref_drop_prob=float(dataset_cfg.get("ref_drop_prob", 0.0)),
        )

        save_root = os.path.join(self.hparams.output_root, self.hparams.experiment_name, 'infer_samples')
        os.makedirs(save_root, exist_ok=True)

        # 断点续推：过滤掉输出已存在的样本（rank0 决策再广播给其它 rank，避免不一致）
        resume_skip = bool(getattr(self.hparams, "resume_skip_existing", True))
        if resume_skip and len(ds.sample_index) > 0:
            kept, skipped = [], 0
            for entry in ds.sample_index:
                out_path = _resolve_save_path(
                    entry["video_path"], dataset_roots, save_root, entry.get("clip_idx", 0)
                )
                if os.path.isfile(out_path):
                    skipped += 1
                else:
                    kept.append(entry)
            if rank == 0:
                print(f"[Infer][resume] kept={len(kept)} skipped(already-exist)={skipped} total={len(ds.sample_index)}", flush=True)
            ds.sample_index = kept

        if len(ds.sample_index) == 0:
            if rank == 0:
                print("[Infer] Nothing to do — all outputs already exist.", flush=True)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        if use_bucket:
            batch_sampler = BucketSampler(
                ds.sample_index,
                batch_size=1,
                shuffle=False,
                drop_last=False,
            )
            dl = torch.utils.data.DataLoader(
                ds,
                batch_sampler=batch_sampler,
                num_workers=dataset_cfg.num_workers,
                pin_memory=dataset_cfg.pin_memory,
            )
        else:
            dl = torch.utils.data.DataLoader(
                ds, batch_size=1, num_workers=dataset_cfg.num_workers,
                drop_last=False, pin_memory=dataset_cfg.pin_memory, shuffle=False
            )

        if self.using_lora:
            ckpt_path = getattr(self.hparams, "ckpt_path", None)
            self._load_lora_from_ckpt(ckpt_path, device)

        pipeline = WanPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            scheduler=self.sample_scheduler,
        )

        if rank == 0:
            print(f"[Infer] Save to: {save_root}  world_size={world_size}", flush=True)

        for batch_idx, batch in enumerate(dl):
            model_input = batch["pixel_values"].to(device)
            model_input2 = batch["pixel_values2"].to(device)
            first_frames = batch.get('first_frames', None)
            if first_frames is not None:
                first_frames = first_frames.to(device).unsqueeze(2)
            prompts = batch["prompts"]

            is_one2three = self.hparams.dataset.is_one2three
            pixel_h = model_input.shape[3]
            pixel_w = model_input.shape[4]

            video_path_field = batch.get("video_path", [None])
            video_path_str = video_path_field[0] if isinstance(video_path_field, (list, tuple)) else video_path_field
            clip_idx_field = batch.get("clip_idx", [0])
            clip_idx_val = int(clip_idx_field[0].item() if hasattr(clip_idx_field[0], "item") else clip_idx_field[0])

            save_path = _resolve_save_path(video_path_str, dataset_roots, save_root, clip_idx_val) \
                if video_path_str else os.path.join(save_root, f"rank{rank}_batch_{batch_idx}_clip{clip_idx_val}.mp4")

            # 二次检查（多 rank 并行下别的 rank 可能已写入；正常 BucketSampler 分片下不会重复）
            if os.path.isfile(save_path):
                print(f"[Infer][rank{rank}] Skip (exists): {save_path}", flush=True)
                continue

            if is_one2three:
                video_gt = model_input2.squeeze(0).permute(1, 0, 2, 3)
                video_gt = ((video_gt + 1) * 0.5).clamp(0, 1)
                video_gt = video_gt.permute(0, 2, 3, 1).detach().cpu().numpy()
                meta = model_input.squeeze(0).permute(1, 0, 2, 3)
                meta = ((meta + 1) * 0.5).clamp(0, 1)
                meta = meta.permute(0, 2, 3, 1).detach().cpu().numpy()

                model_input_lat = self.vae.encode(model_input).latent_dist.sample()
                model_input_lat = (model_input_lat - self.latents_mean) / self.latents_std

                if first_frames is None:
                    first_frames_lat = model_input_lat[:, :, :1].detach() * 0
                else:
                    first_frames_lat = self.vae.encode(first_frames).latent_dist.sample()
                    first_frames_lat = (first_frames_lat - self.latents_mean) / self.latents_std

                attention_kwargs = {
                    'encoder_contion_states': model_input_lat,
                    'encoder_first_states': first_frames_lat,
                }
            else:
                video_gt = model_input.squeeze(0).permute(1, 0, 2, 3)
                video_gt = ((video_gt + 1) * 0.5).clamp(0, 1)
                video_gt = video_gt.permute(0, 2, 3, 1).detach().cpu().numpy()
                meta = model_input2.squeeze(0).permute(1, 0, 2, 3)
                meta = ((meta + 1) * 0.5).clamp(0, 1)
                meta = meta.permute(0, 2, 3, 1).detach().cpu().numpy()

                model_input2_lat = self.vae.encode(model_input2).latent_dist.sample()
                model_input2_lat = (model_input2_lat - self.latents_mean) / self.latents_std
                attention_kwargs = {
                    'encoder_contion_states': model_input2_lat,
                }

            _ = self.encode_prompt(prompts, device=device)

            out = pipeline(
                prompt=prompts,
                height=pixel_h,
                width=pixel_w,
                num_frames=self.hparams.dataset.sample_n_frames,
                guidance_scale=5.0,
                attention_kwargs=attention_kwargs,
            )
            video_generate = out.frames[0]

            meta, video_generate, video_gt = self._align_frames(meta, video_generate, video_gt, is_one2three)
            concat = np.concatenate([meta, video_generate, video_gt], axis=1)

            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            tmp_path = save_path + f".tmp.rank{rank}.mp4"
            export_to_video(concat, output_video_path=tmp_path, fps=self.hparams.dataset.fps)
            os.replace(tmp_path, save_path)
            print(f"[Infer][rank{rank}] Saved: {save_path}", flush=True)

        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/inference_config.yaml", help="path to the yaml config file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_path", type=str, default="", help="path to a .ckpt with LoRA/extra weights")
    args, extras = parser.parse_known_args()
    args = vars(args)

    opt = OmegaConf.merge(
        OmegaConf.load(args['config']),
        OmegaConf.from_cli(extras),
        OmegaConf.create(args),
        OmegaConf.create({"num_nodes": int(os.environ.get("NUM_NODES", 1))}),
        OmegaConf.create({"num_gpus": int(torch.cuda.device_count())}),
    )

    opt.ckpt_path = None if args['ckpt_path'] in ("", "null", "None") else args['ckpt_path']

    L.seed_everything(opt.seed)

    system = InteractionVideoSystemInfer(opt)
    system.run_infer()


if __name__ == "__main__":
    main()
