# Callbacks

Training callbacks for FastGen. All callbacks inherit from `Callback` and hook into the training lifecycle.

## Available Callbacks

| Callback | Description |
|----------|-------------|
| [`EMACallback`](ema.py) | Exponential Moving Average updates (constant, power, or halflife schedules) |
| [`GradClipCallback`](grad_clip.py) | Gradient norm clipping with NaN/Inf handling |
| [`WandbCallback`](wandb.py) | Weights & Biases logging (metrics, images, videos) |
| [`GPUStatsCallback`](gpu_stats.py) | GPU memory and utilization logging |
| [`MemTrackerCallback`](gpu_mem_profiler.py) | GPU memory profiling with HTML visualizations |
| [`TrainProfilerCallback`](train_profiler.py) | Training speed and timing breakdown |
| [`ParamCountCallback`](param_count.py) | Parameter count logging |
| [`ForcedWeightNormCallback`](forced_weight_norm.py) | Forced weight normalization (EDM2) |
| [`CTScheduleCallback`](ct_schedule.py) | Consistency training curriculum schedule |

## Usage

Configure callbacks as a dictionary in the config:

```python
from fastgen.configs.callbacks import WANDB_CALLBACK, GradClip_CALLBACK, EMA_CALLBACK
from omegaconf import DictConfig

config.trainer.callbacks = DictConfig({
    **WANDB_CALLBACK,
    **GradClip_CALLBACK,
    **EMA_CALLBACK,
})
```

Predefined configs are in `fastgen/configs/callbacks.py`.

## Lifecycle Hooks

Callbacks can override these hooks:

| Phase | Hooks |
|-------|-------|
| Setup | `on_app_begin`, `on_model_init_start/end`, `on_optimizer_init_start/end`, `on_load_checkpoint_start/end`, `on_dataloader_init_start/end` |
| Training | `on_train_begin/end`, `on_training_step_begin/end`, `on_training_accum_step_begin`, `on_backward_begin`, `on_optimizer_step_begin` |
| Validation | `on_validation_begin/end`, `on_validation_step_begin/end` |
| Checkpointing | `on_save_checkpoint_start/success/end`, `state_dict`, `load_state_dict` |
| Cleanup | `on_app_end` |

## Custom Callbacks

```python
from fastgen.callbacks.callback import Callback

class MyCallback(Callback):
    def on_training_step_end(self, model, data_batch, output_batch, loss_dict, iteration=0):
        if iteration % self.config.trainer.logging_iter == 0:
            print(f"Step {iteration}: loss = {loss_dict['total_loss'].item():.4f}")
```

Callbacks have access to `self.config` and `self.trainer` after initialization.
