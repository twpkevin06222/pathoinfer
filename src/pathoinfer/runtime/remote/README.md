# pathoinfer - runtime/remote

Clients for models served by another process: Triton Inference Server for tile models
and a vLLM server for vision-language models (VLMs). These modules only speak the
servers' network protocols. They never import `vllm` or Triton's server code, never
start or stop servers, and need none of their dependencies. (Offline vLLM generation in
this process is `runtime/local/vllm_offline.py`, in its own venv.)

| Module | Status | Extra | Talks to |
|---|---|---|---|
| `vllm.py` | `VllmClient`: questions about images, greedy, with retries, timeouts and a concurrency limit | `[vllm-client]` (httpx) | Any OpenAI-compatible server, normally `vllm serve` |
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

Print the server command for a model folder (bf16, or one written by
`pathoinfer convert vlm`) and run it:

```bash
pathoinfer serve vllm models/pvqa-fp8 --config vllm.yaml
# vllm serve models/pvqa-fp8 --host 0.0.0.0 --port 8000 --served-model-name pvqa \
#   --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.9 \
#   --tensor-parallel-size 1 --limit-mm-per-prompt '{"image": 1}' --quantization compressed-tensors
```

`serving/vllm.py` derives the command from the card: `dtype` from `precision` (quantised
cards keep the vision encoder in bf16), `quantization` from the folder's
`quantization_config` (a card and weights that disagree are refused),
`--mm-processor-kwargs '{"max_pixels": ...}'` from `input.max_pixels`, and one image per
prompt. `VllmServeConfig` (the YAML) sets `gpu_memory_utilization`, `max_model_len`
(default 8192: a crop question needs far less than the model's maximum),
`max_num_seqs`, `tensor_parallel_size`, `served_model_name` (default: the card's name),
`allowed_local_media_path`, `host` and `port`.

Then ask it about crops:

```python
from pathoinfer.runtime.remote.vllm import VllmClient

client = VllmClient("http://localhost:8000", concurrency=16)
answer = client.ask("crops/slide_name-01_0042.png", "Is there tumour in this image?")
answers = client.ask_many([(crop, "Is there tumour?") for crop in crops])  # in order
client.close()
```

- `VllmClient(base_url, model=None, *, timeout_s=120, retries=3, backoff_s=1.0,
  concurrency=8, local_media=False, api_key=None, transport=None)`. Without `model` it
  asks `GET /v1/models` and uses the first one served. `transport` replaces the HTTP
  transport; tests pass `httpx.MockTransport`.
- `ask(image, question, max_tokens=64, *, system=None, json_schema=None)` posts one
  user message to `POST /v1/chat/completions`: the image, then the question.
  `temperature` is 0, so answers are reproducible, which parity relies on. `json_schema`
  sets `response_format`, so vLLM constrains the answer to that schema.
- Images are files or RGB arrays. They go as base64 data URLs (arrays as PNG), or with
  `local_media=True` as `file://` URIs, which skips encoding but needs a server started
  with `--allowed-local-media-path` that sees the same paths.
- At most `concurrency` requests are in flight per client, from any number of threads;
  vLLM batches concurrent requests itself. `ask_many` runs `concurrency` at a time.
- Connection errors, timeouts and HTTP 408, 429 and 5xx are retried `retries` times,
  waiting `backoff_s`, then twice as long each time, with a warning per retry. Other
  errors raise `httpx.HTTPStatusError` at once.

Crops should be made the way the model was fine-tuned: the `VqaCard`'s `input.tile_size`
and `input.mpp`. Boxes in grounding answers become level-0 px with
`runtime.vlm.parse_boxes(answer, card.output, Crop(x, y, width, height, image_size))`,
where the crop is the level-0 region the image was read from.

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

`tests/runtime/test_vllm_client.py` runs `VllmClient` against a mocked server
(`httpx.MockTransport`): the request (model from `/v1/models`, image as a data URL or
`file://` URI, `temperature: 0`, `response_format`, the API key), retries of 5xx, 429
and timeouts, no retry of 4xx, answers in order, and the concurrency limit across
threads. `tests/serving/test_vllm.py` checks the launch arguments, and in a venv with vLLM
that each is an `EngineArgs` field and a `vllm serve` flag. A live server is exercised only by
`tests/convert/test_vlm_full.py` (markers `gpu` and `slow`) when
`PATHOINFER_VLLM_REFERENCE_URL` and `PATHOINFER_VLLM_CANDIDATE_URL` are set.
