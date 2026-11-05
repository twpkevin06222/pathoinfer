# pathoinfer - runtime

Runs one model on one batch of tiles: tensors in, tensors out. A runtime knows nothing
about slides; the engine handles tiling, stitching and writing.

## Contract

| | |
|---|---|
| Input | `(N, H, W, 3)` RGB `uint8` torch tensor, any `N`. CPU, pinned or on the GPU (what `tiling.dataset` yields). |
| Output | `dict[str, torch.Tensor]` keyed by `card.output.tensor_names()`, `N` rows each, on `runtime.device`. Raw model outputs: no softmax, thresholding or NMS. |
| Protocol | `Runtime` in `base.py`: `card`, `device`, `batch_size`, `predict(batch)`, `warmup()`. |

## Usage

```python
import torch

from pathoinfer.config import RuntimeConfig
from pathoinfer.registry.loader import load_model
from pathoinfer.runtime.factory import make_runtime

card, model_dir = load_model("models/tumour-seg")  # any source the loader accepts
runtime = make_runtime(card, model_dir, RuntimeConfig(device="auto", batch_size=32))
runtime.warmup()

tiles = torch.zeros((50, card.input.tile_size, card.input.tile_size, 3), dtype=torch.uint8)
outputs = runtime.predict(tiles)  # {"logits": (50, n_classes, H, W)}
```

`RuntimeConfig` is the `runtime:` section of a run config (`configs/*.yaml`):

| Field | Default | Meaning |
|---|---|---|
| `device` | `auto` | `auto` (CUDA when available, else CPU), `cpu`, `cuda` or `cuda:N` |
| `batch_size` | `32` | Fixed batch size every backend sees |
| `tensorrt` | `true` | ONNX models use the TensorRT EP when its library loads |
| `autocast` | `null` | PyTorch models only: `bf16` or `fp16` |

## Preprocessing

`base.preprocess` is the only preprocessing in pathoinfer, and it reads only
`card.input`, so every runtime feeds a model what training did:

1. Move the `uint8` batch to the device (4x fewer bytes than float).
2. Reorder channels to `channel_order` (`RGB` or `BGR`).
3. Unless the input dtype is `uint8`: `(x * scale - mean) / std` per channel.
4. Lay out as `layout` (`NCHW` or `NHWC`) and cast to `dtype` (`float32`, `float16`).

## Fixed batches

`base.run_in_fixed_batches` cuts `predict`'s input into `batch_size` chunks. The last
chunk is padded with zero tiles and the padding removed from the outputs, so a backend
only ever sees one input shape (TensorRT, CUDA graphs) and padding never reaches callers.

## Backends

| Module | Framework | Notes |
|---|---|---|
| `local/onnx.py` | `onnx` | Needs the `[onnx]` extra. Providers: TensorRT EP if `libnvinfer.so.10` loads, then CUDA, then CPU. On CUDA, input and outputs stay on the GPU (IO binding + DLPack). The graph's input dtype must equal `card.input.dtype`, so fp16 graphs get fp16 inputs. |
| `local/torch.py` | `torch` | `torch.export` (`.pt2`, preferred) or TorchScript (`.pt`, `.pth`, `.ts`; deprecated by torch 2.14). `inference_mode`, optional bf16 / fp16 autocast. |
| `local/tensorrt.py` | `tensorrt` | Stub; TensorRT plans land in v0.2. |
| `remote/triton.py` | | Stub; Triton gRPC client, v0.3. |
| `remote/vllm.py` | | Stub; OpenAI-compatible HTTP client for VLMs. Never imports `vllm`. |

`factory.make_runtime` picks the backend from `card.framework`. A `safetensors` card is
rejected: it holds weights only.

## Testing and benchmarks

- `tests/runtime/test_base.py` checks preprocessing against hand-computed numbers and
  the padding behaviour.
- `tests/runtime/test_local.py` runs the fixture model through every available runtime
  (CPU, CUDA, fp16, autocast) and compares each with ONNX on CPU within a tolerance.
- `scripts/bench_infer.py MODEL_DIR...` prints tiles/s per runtime and precision.
