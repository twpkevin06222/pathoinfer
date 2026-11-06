# pathoinfer - deploy

Containers for the slide job API (`pathoinfer serve api`) with models served by Triton
Inference Server, and a demo that runs one job end to end on a synthetic slide.

| File | What it is |
|---|---|
| `Dockerfile` | The pathoinfer image: the job API and the CLI, plus the demo model, slide and Triton repository. CPU by default; `--build-arg TORCH_INDEX=cu130` for CUDA |
| `Dockerfile.triton` | Triton 26.10 with the demo repository baked in |
| `docker-compose.yml` | `triton`, `api` and a `demo` client |
| `docker-compose.gpu.yml` | Override that gives Triton the GPU |
| `demo/setup.py` | Writes the synthetic slide and the Triton repository (at image build time) |
| `demo/run_job.py` | Submits a job, follows it, checks its outputs (standard library only) |

## Run the demo

From the repository root:

```bash
docker compose -f deploy/docker-compose.yml up --build --exit-code-from demo
```

The first run downloads the Triton image (about 9.4 GB compressed) and builds two
images. Building the pathoinfer image writes the demo's synthetic slide
(`/opt/pathoinfer/demo/data/slides/synthetic.tiff`, 4096 x 3072 px at 0.25 µm/px) and a
Triton repository for the demo model (`tiny-dark-onnx`: class 1 wherever a pixel is
darker than glass); the demo Triton image copies that repository to `/models`. Then:

1. **triton** serves `/models`; it is healthy once the model is ready.
2. **api** runs the job API on port 8080 with `PATHOINFER_TRITON_URL=triton:8001`, so
   jobs run their model on Triton. The API reads the model folder only for its card.
3. **demo** submits a segmentation job on the synthetic slide, prints its progress and
   outputs, and checks the GeoJSON has polygons. With `--exit-code-from demo`, compose
   stops everything and exits with the demo's status: 0 means the job ran end to end.

The demo's data is made at build time rather than by a one-shot `setup` service on
purpose: Compose's `--exit-code-from` does not return when the stack holds a one-shot
service that has already exited (seen with Compose 5.2).

GPU is optional. To serve the model on the GPU:

```bash
docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml up --build --exit-code-from demo
```

To keep the stack running for your own jobs, leave out `--exit-code-from demo` and use
the API at `http://localhost:8080` (interactive docs at `/docs`):

```bash
curl -X POST localhost:8080/jobs -H 'Content-Type: application/json' -d '{
  "slide": "/opt/pathoinfer/demo/data/slides/synthetic.tiff",
  "config": {"task": "segmentation", "model": "/opt/pathoinfer/demo/model"}}'
curl localhost:8080/jobs/JOB_ID
curl localhost:8080/jobs/JOB_ID/outputs
```

## The job API

| Endpoint | Does |
|---|---|
| `POST /jobs` | `{"slide": PATH_OR_URI, "config": RUN_CONFIG}`; `202` with the job. A bad config or missing slide is a `422` |
| `GET /jobs`, `GET /jobs/{id}` | Status (`queued`, `running`, `done`, `failed`, `cancelled`), progress in tiles, error |
| `GET /jobs/{id}/outputs` | Names, sizes and URLs of a finished job's files |
| `GET /jobs/{id}/outputs/{name}` | One file |
| `POST /jobs/{id}/cancel` | Dequeue a waiting job or stop a running one |
| `GET /health` | Liveness, queued and running counts |

- **Slides** are paths inside the container (mount them), `file://` URIs, or `http(s)://`
  URLs, which the job downloads first.
- **Config** is a run config (`examples/*.yaml`, as JSON) without `slides`,
  `outputs.dir` and `resume`, which the server sets. `model` is a model folder inside the
  container. Outputs are written per job under `/data/jobs/<id>/outputs/`.
- **Concurrency.** Each job runs the engine in its own process; `--max-jobs` (default 1)
  run at once and the rest wait in order. One job per GPU is usual: a job already
  overlaps reading, inference and stitching.
- **Cancellation** removes a waiting job, or stops a running one's process (and its
  reader processes).
- **State** is on disk under `--jobs-dir`, so jobs survive an API restart; a job that was
  running when the API stopped is marked failed.

## Serving your own models

Write a Triton repository from model folders (ONNX models or TensorRT plans), run the
official image (`nvcr.io/nvidia/tritonserver:26.10-py3`) with it mounted at `/models`,
and mount the model folders into `api` for their cards:

```bash
pathoinfer serve triton-repo models/tumour-seg models/cells --repository model_repository
```

```yaml
# triton_repo.yaml (TritonRepoConfig), passed with --config
max_batch_size: 32                # at least the batch_size jobs send
preferred_batch_sizes: [8, 16, 32]
max_queue_delay_us: 2000
instances: 1
kind: auto                        # auto (GPU when there is one), gpu or cpu
```

Each model is served under its card's name, which is the name the API's Triton runtime
asks for. TensorRT plans only run on the GPU architecture and TensorRT version they were
built with, so build them (`pathoinfer convert trt`) with the TensorRT release in the
Triton image you deploy (26.10 here).

Without Triton, run the API alone (`docker run -p 8080:8080 -v ...:/data pathoinfer:cpu`):
jobs then run their models in the job process, as `pathoinfer run` does.

## Notes

- The image installs patchology from GitHub at the commit CI tests
  (`PATCHOLOGY_REF`); bump both together.
- `shm_size` matters for real slides: DataLoader reader processes pass tiles through
  shared memory, which Docker limits to 64 MB by default.
- The containers run as a non-root user (uid 1000); mounted folders must be readable by
  it, and `/data` writable.
