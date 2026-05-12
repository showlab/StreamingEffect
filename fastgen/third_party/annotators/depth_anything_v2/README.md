## Depth Anything V2

The VACE WAN model requires pretrained weights for Depth Anything V2 to extract depth information from video frames.

### Download Instructions

Download the pretrained weights from the [official repository](https://github.com/DepthAnything/Depth-Anything-V2) and place them in `${CKPT_ROOT_DIR}/annotators/`:

```bash
mkdir -p ${CKPT_ROOT_DIR}/annotators
wget -O ${CKPT_ROOT_DIR}/annotators/depth_anything_v2_vitl.pth \
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
```

### Configuration

The experiment configs use `CKPT_ROOT_DIR` by default:

```python
config.model.net.depth_model_path = f"{CKPT_ROOT_DIR}/annotators/depth_anything_v2_vitl.pth"
```

You can override `CKPT_ROOT_DIR` via environment variable or specify a custom path directly.

### Notes

- The depth model is used by VACE (Visibility-Aware Context Encoding) to enhance video generation
- The model will be loaded automatically when running VACE WAN training or inference
- Ensure the weights file has proper read permissions

The module is imported by `networks/VaceWan/modules/vace_depth_annotator.py` and is essential for VACE WAN functionality.


### License

See [`licenses/depth_anything_v2/LICENSE`](../../../../licenses/depth_anything_v2/LICENSE) for details.