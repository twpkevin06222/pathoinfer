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

`RuntimeConfig` is the `runtime:` section of a run config (`examples/*.yaml`):

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
| `local/tensorrt.py` | `tensorrt` | TensorRT plans (`pathoinfer convert trt`), built per GPU and TensorRT version; needs `[trt]`. |
| `remote/triton.py` | any | `TritonRuntime`: the protocol over gRPC to Triton (`runtime.triton_url`); preprocessing stays here. See `remote/README.md`. |

`factory.make_runtime` picks the backend from `card.framework`. A `safetensors` card is
rejected: it holds weights only.

## VLMs

Vision-language models (`VqaCard`) answer questions about slide crops instead of
returning tensors, so they sit beside the `Runtime` protocol rather than inside it.
Both ways of running them build the same chat message (`vlm.chat_messages`) and decode
greedily, so a model answers alike either way:

| Module | Runs | Install |
|---|---|---|
| `remote/vllm.py` | `VllmClient`: HTTP to a `vllm serve` server, with retries, timeouts and a concurrency limit. The way to serve. | `[vllm-client]` (httpx) |
| `local/vllm_offline.py` | `OfflineVlm`: vLLM's Python engine in this process, for batch evaluation and captioning. The only module that imports `vllm`. | `[vllm]`, in its own venv |
| `vlm.py` | Shared: images as data URLs or `file://` URIs, the chat message, and `parse_boxes`, which maps boxes in an answer to level-0 px through the `Crop` the image came from. | core |

`serving/vllm.py` (`pathoinfer serve vllm MODEL`) builds the server's arguments from the
card; `OfflineVlm` passes the same ones to `vllm.LLM`. The grounding chain these build
toward (Qwen3-VL boxes, then SAM 3 masks) is in `docs/vlm_grounding.md`.

### The vLLM venv

vLLM 0.33 pins torch 2.14.0, numpy below 2.5 (through numba), transformers below 5.21 and
compressed-tensors 0.17. Those clash with `[vlm-convert]` (llm-compressor pins
compressed-tensors 0.20.1), and on Python 3.12 its mistral-common caps numpy below 2.4.
So `[vllm]` is not in `[all]`, needs Python 3.13+, and lives in a venv of its own.
The workspace lock resolves it as a separate fork (`[tool.uv] conflicts` in
`pyproject.toml`), so the main venv keeps torch 2.14.1 and numpy 2.5.4:

```bash
cd packages/pathoinfer
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --python 3.13 --extra vllm
.venv-vllm/bin/python -c "from pathoinfer.runtime.local.vllm_offline import OfflineVlm"
```

```python
from pathoinfer.runtime.local.vllm_offline import OfflineVlm
from pathoinfer.serving.vllm import VllmServeConfig

vlm = OfflineVlm("models/pvqa-fp8", VllmServeConfig(max_model_len=4096))
answers = vlm.generate([("crops/slide_name-01_0042.png", "Is there tumour?")], max_tokens=16)
```

## Testing and benchmarks

- `tests/runtime/test_base.py` checks preprocessing against hand-computed numbers and
  the padding behaviour.
- `tests/runtime/test_local.py` runs the fixture model through every available runtime
  (CPU, CUDA, fp16, autocast) and compares each with ONNX on CPU within a tolerance.
- `scripts/bench_infer.py MODEL_DIR...` prints tiles/s per runtime and precision.
- `tests/runtime/test_vlm.py` pins `parse_boxes` (every unit and format to the same
  level-0 box, clipping, JSON and text answers) and the image URLs;
  `test_vllm_client.py` runs the client against a mocked server (retries, timeouts,
  order, the concurrency limit); `test_vllm_offline.py` drives `OfflineVlm` through a
  stand-in `vllm` module, and checks it refuses without vllm.
