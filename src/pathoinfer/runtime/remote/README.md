# pathoinfer - runtime/remote

Clients for models served by another process: Triton Inference Server for tile models
and a vLLM server for vision-language models (VLMs). pathoinfer only speaks the
servers' network protocols. It never imports `vllm` or Triton's server code, never starts
or stops servers, and needs none of their dependencies.

| Module | Status | Extra | Talks to |
|---|---|---|---|
| `vllm.py` | `VllmClient`: one question about one image, greedy | `[vllm-client]` (httpx) | Any OpenAI-compatible server, normally `vllm serve` |
| `triton.py` | `TritonRuntime`: the `Runtime` protocol over gRPC | `[triton]` (tritonclient) | Triton Inference Server |

`TritonRuntime` implements the same `Runtime` protocol as the local runtimes
(`runtime/base.py`), so the engine runs a slide the same way in-process or against a
server. `VllmClient` answers one question at a time; the slide-level VLM runtime (crops,
questions, aggregation) will build on it.

## Why the servers run separately

vLLM pins its own torch, transformers and CUDA builds, and this workspace is held at
other versions (llm-compressor caps torch at 2.14.1 and transformers at 5.21.0), so the
two cannot share an environment. A GPU server also needs its own lifecycle: a minute or
more of startup, memory sizing, ports, restarts and scaling. Docker, Kubernetes or
systemd handle that; pathoinfer connects by URL.

Pin the vLLM version you deploy. A quantised folder's format (compressed-tensors from
`pathoinfer convert vlm`) must be one that version serves.

## vLLM (`vllm.py`)

Start a server for a model folder (bf16, or one written by `pathoinfer convert vlm`):

```bash
vllm serve models/pvqa-fp8 --port 8000
```

Then ask it about a crop:

```python
from pathoinfer.runtime.remote.vllm import VllmClient

client = VllmClient("http://localhost:8000")  # model: the server's first, or model="..."
answer = client.ask("crops/slide_name-01_0042.png", "Is there tumour in this image?")
client.close()
```

- `VllmClient(base_url, model=None, timeout_s=120.0, transport=None)`. Without `model` it
  asks `GET /v1/models` and uses the first one served. `transport` replaces the HTTP
  transport; tests pass `httpx.MockTransport`.
- `ask(image, question, max_tokens=64)` posts one user message to
  `POST /v1/chat/completions`: the image as a base64 data URL, then the question as text.
  `temperature` is 0, so answers are reproducible, which parity relies on. It returns the
  answer text and raises `httpx.HTTPStatusError` on an error response.
- One request per call. Run calls from several threads for throughput: vLLM batches
  concurrent requests itself. `convert vlm-parity` uses `concurrency` threads per server.

Crops should be made the way the model was fine-tuned: the `VqaCard`'s `input.tile_size`
and `input.mpp`.

Used by `convert.vlm_parity.gate`, which asks a bf16 server and a quantised one the same
eval questions (`convert/README.md`).

## Triton (`triton.py`)

Set `runtime.triton_url` in a run config (or `PATHOINFER_TRITON_URL` for the job API) and
`make_runtime` returns a `TritonRuntime` instead of a local runtime:

```yaml
runtime:
  triton_url: triton:8001       # gRPC
  triton_model: null            # default: the card's name
  batch_size: 32                # at most the model's max_batch_size on the server
```

- The model folder is still read, for its card: preprocessing (normalisation, layout,
  dtype) runs here on the CPU exactly as for the local runtimes, and the stitcher needs
  the card's outputs and classes.
- Each fixed-size batch (the last one padded) is one gRPC inference request for the
  card's output tensors; outputs come back as CPU tensors. Triton's dynamic batching
  merges requests from concurrent jobs.
- At start it checks the model is ready and its input name and dtype match the card,
  and raises `ValueError` if not.
- `serving/triton_repo.py` (`pathoinfer serve triton-repo`) writes the repository that
  serves a model folder under its card's name; `deploy/` runs it with Docker Compose.

## Testing

`tests/serving/test_triton.py` checks the repository generator against Triton's own config
schema and, with `PATHOINFER_TRITON_URL` set to a server serving `tiny_dark_onnx`
(markers `gpu` and `slow`), that `TritonRuntime` matches the local ONNX runtime and that
a whole slide gives the same result through Triton as locally.

`tests/convert/test_vlm.py::test_vllm_client_sends_the_image_and_greedy_settings` runs
`VllmClient` against `httpx.MockTransport`: the model name comes from `/v1/models`, the
request carries the image as a PNG data URL, the question and `temperature: 0`, and the
answer text comes back. A live server is exercised only by
`tests/convert/test_vlm_full.py` (markers `gpu` and `slow`) when
`PATHOINFER_VLLM_REFERENCE_URL` and `PATHOINFER_VLLM_CANDIDATE_URL` are set.
