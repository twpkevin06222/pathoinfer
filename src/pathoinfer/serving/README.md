# pathoinfer - serving

Serving runs pathoinfer for other programs over the network. It wraps the engine; it
does not change what a run computes. A slide served through the job API gives the same
outputs as `pathoinfer run` with the same config.

There are three kinds of serving:

| Module | Serves | Start it with | Extra |
|---|---|---|---|
| `api.py` | Whole slides, as jobs: submit a slide and a run config, poll, download the outputs | `pathoinfer serve api` | `[serve]` |
| `triton_repo.py` | Tile models (ONNX or TensorRT plans) on NVIDIA Triton, so many jobs share one GPU server | `pathoinfer serve triton-repo`, then Triton | `[triton]` for the client |
| `vllm.py` | VLMs (`VqaCard` folders) on a vLLM server | `pathoinfer serve vllm` prints the `vllm serve` command | `[vllm-client]` for the client |

```text
client --HTTP--> job API --one process per job--> engine (run_slide)
                                                    |
                                                    +-- model in-process (ONNX Runtime, PyTorch, TensorRT)
                                                    +-- or TritonRuntime --gRPC--> Triton (tile models)

client / VllmClient --HTTP (OpenAI API)--> vLLM (VLMs)
```

Preprocessing always runs on the client side (`runtime.base.preprocess`), so a model gives
the same output in-process and on Triton. Docker images and a Compose stack that runs
the API with Triton are in [`deploy/README.md`](../../../deploy/README.md).

## Slide job API

```bash
pathoinfer serve api --port 8080 --jobs-dir jobs --max-jobs 1
```

Submit a slide and a run config (as `examples/*.yaml`, written as JSON, without `slides`,
`outputs.dir` and `resume`, which the server sets):

```bash
curl -X POST localhost:8080/jobs -H 'Content-Type: application/json' -d '{
  "slide": "/data/slides/slide_name-01.svs",
  "config": {"task": "segmentation", "model": "/models/tumour-seg"}
}'
# 202 {"id": "3f2c...", "status": "queued", ...}

curl localhost:8080/jobs/3f2c...                 # status and progress in tiles
curl localhost:8080/jobs/3f2c.../outputs         # names, sizes and URLs when done
curl -O localhost:8080/jobs/3f2c.../outputs/slide_name-01.geojson.gz
```

| Endpoint | Does |
|---|---|
| `POST /jobs` | Submit; `202` with the job. A bad config or missing slide is a `422` |
| `GET /jobs`, `GET /jobs/{id}` | Status (`queued`, `running`, `done`, `failed`, `cancelled`), progress, error |
| `GET /jobs/{id}/outputs`, `/outputs/{name}` | A finished job's files |
| `POST /jobs/{id}/cancel` | Dequeue a waiting job or stop a running one |
| `GET /health` | Liveness, queued and running counts |

- **Slides** are paths on the server, `file://` URIs, or `http(s)://` URLs that the job
  downloads first.
- **Jobs** run the engine in their own process, `--max-jobs` at a time; the rest wait in
  order. One job per GPU is usual, because a job already overlaps reading, inference and
  stitching. Cancelling stops the job's process and its reader processes.
- **State** lives on disk under `--jobs-dir`: `<id>/job.json` (written only by the API
  process), `<id>/progress.json` and `<id>/result.json` (written only by the job), and
  `<id>/outputs/`. Jobs survive an API restart; a job that was running when the API
  stopped is marked failed, and queued jobs are queued again.
- **Triton.** With `PATHOINFER_TRITON_URL` set, jobs whose config names no
  `runtime.triton_url` run their model on that Triton server. The model folder is still
  read for its card.

## Triton model repository

```bash
pathoinfer serve triton-repo models/tumour-seg models/cells --repository model_repository \
  --config triton_repo.yaml
docker run --gpus all -p 8001:8001 -v $PWD/model_repository:/models \
  nvcr.io/nvidia/tritonserver:26.10-py3 tritonserver --model-repository=/models
```

Each model is written as `<card name>/config.pbtxt` and `<card name>/1/model.onnx` (or
`model.plan`), and served under its card's name, the name `TritonRuntime` asks for.
`config.pbtxt` is built as Triton's own `ModelConfig` message, so it holds only fields
Triton knows.

```yaml
# triton_repo.yaml (TritonRepoConfig)
max_batch_size: 32                # at least the runtime.batch_size clients send
preferred_batch_sizes: [8, 16, 32]
max_queue_delay_us: 2000          # how long Triton waits to fill a batch
instances: 1                      # model copies
kind: auto                        # auto (GPU when there is one), gpu or cpu
```

A run uses Triton with `runtime.triton_url` (gRPC, for example `localhost:8001`) in its
config. TensorRT plans run only on the GPU architecture and TensorRT version they were
built with: build them (`pathoinfer convert trt`) with the TensorRT release of the Triton
image you deploy (26.10 here).

## vLLM for VLMs

```bash
pathoinfer serve vllm models/pvqa-qwen3vl --config vllm.yaml
# vllm serve models/pvqa-qwen3vl --host 0.0.0.0 --port 8000 --served-model-name path-vqa --dtype bfloat16 ...
```

The command only prints the `vllm serve` command line; run it in the vLLM venv (vLLM is
not installed with pathoinfer, see `runtime/README.md`). dtype, quantisation, image size
(`max_pixels`) and one image per prompt come from the model card and the folder's
`config.json`; a card and weights that disagree are refused.

```yaml
# vllm.yaml (VllmServeConfig)
gpu_memory_utilization: 0.9
max_model_len: 8192
max_num_seqs: null                # vLLM's default
tensor_parallel_size: 1
served_model_name: null           # default: the card's name
allowed_local_media_path: null    # a folder, to send file:// images instead of base64
host: 0.0.0.0
port: 8000
```

Clients talk to the server with `runtime.remote.vllm.VllmClient`. Offline batch runs use
the same engine arguments through `runtime.local.vllm_offline.OfflineVlm`. How answers
become boxes on the slide is in `docs/vlm_grounding.md`.
