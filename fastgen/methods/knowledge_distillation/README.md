# Knowledge Distillation Methods

Learn from pre-computed teacher denoising trajectories.

## KD (Knowledge Distillation)

**File:** [`KD.py`](KD.py) | **Reference:** [Luhman & Luhman, 2021](https://arxiv.org/abs/2101.02388)

MSE loss between student prediction and teacher output: `L = ||f(x_t, t) - x_0^teacher||²`

**Data Requirements:**
- Single-step: `{"real": clean, "noise": noise, "condition": cond}`
- Multi-step: `{"real": clean, "path": [B, steps, C, H, W], "condition": cond}`

**Key Parameters:**
- `student_sample_steps`: Number of student steps
- `sample_t_cfg.t_list`: Timesteps (must align with path)

**Configs:** [`WanT2V/config_kd.py`](../../configs/experiments/WanT2V/config_kd.py), [`SDXL/config_kd.py`](../../configs/experiments/SDXL/config_kd.py), [`CogVideoX/config_kd.py`](../../configs/experiments/CogVideoX/config_kd.py)

---

## CausalKD

**File:** [`KD.py`](KD.py) | **Reference:** [Yin et al., 2024](https://arxiv.org/abs/2412.07772)

KD for causal video models with inhomogeneous timesteps and autoregressive generation.

**Data Requirements:**
- `{"real": [B,C,T,H,W], "path": [B,steps,C,T,H,W], "condition": cond}`

**Key Parameters:**
- `context_noise`: Noise for cached context
- See also key parameters of KD above

**Configs:** [`WanT2V/config_kd_path.py`](../../configs/experiments/WanT2V/config_kd_path.py)
