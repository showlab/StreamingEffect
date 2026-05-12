## WAN 2.1 Prompt Expansion

Expands short prompts into detailed descriptions optimized for WAN video generation using Qwen language models.

### Usage in Inference

Enable prompt expansion via command-line arguments in `scripts/inference/video_model_inference.py`:

```bash
python scripts/inference/video_model_inference.py \
    --prompt_expand_model Qwen2.5_14B \
    --prompt_expand_model_seed 42 \
    ...
```

**Arguments:**
- `--prompt_expand_model`: Qwen model to use (`Qwen2.5_3B`, `Qwen2.5_7B`, or `Qwen2.5_14B`)
- `--prompt_expand_model_seed`: Random seed for reproducible expansion (default: 0)

When enabled, prompts are expanded on rank 0 and broadcast to all ranks. The expanded prompt includes subject details, visual style, camera angles, and motion attributes.

### Source

Code from [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1) at commit [7c81b2f](https://github.com/Wan-Video/Wan2.1/commit/7c81b2f27defa56c7e627a4b6717c8f2292eee58). Extracted prompt expansion from `wan/utils` and removed dashscope dependency.

### License

See [`licenses/Wan/LICENSE`](../../../licenses/Wan/LICENSE) for details.
