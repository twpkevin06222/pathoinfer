# pathoinfer - engine

Processes slides: for each slide of a run config, builds the tissue grid, reads tiles in
worker processes, runs the model, stitches on a separate thread and writes the outputs.
The engine knows nothing about how the model executes (that is `runtime/`) or how
outputs are encoded (that is `io/`).

## Usage

From the command line:

```bash
pathoinfer validate examples/segmentation.yaml   # config, model and slides; runs nothing
pathoinfer run examples/segmentation.yaml        # exit 1 if any slide failed
```

From Python:

```python
from pathlib import Path

from pathoinfer.config import load_config
from pathoinfer.engine.pipeline import load, run_batch, run_slide

cfg = load_config("examples/segmentation.yaml")
results = run_batch(cfg)  # every slide of cfg.slides

model = load(cfg)  # or one slide, reusing a loaded model
result = run_slide(cfg, Path("slides/slide_name-01.svs"), model)
print(result.status, result.tiles, result.objects, result.wall_s)
```

`examples/*.yaml` documents every config field; `docs/outputs.md` every output.

## Contract

| | |
|---|---|
| `load(cfg)` | Loads the model (`registry.loader.load_model`), checks its task matches the config's, builds and warms up the runtime. Returns a `Model` with the settings hash. Raises on a task mismatch; classification raises `NotImplementedError` until v0.2. |
| `run_slide(cfg, path, model=None)` | Runs one slide and writes its outputs. Raises on any failure; nothing is marked finished. |
| `run_batch(cfg)` | Runs every slide, loading the model once. A failing slide is logged and recorded; the batch carries on. Returns one `RunResult` per slide. Raises only on problems with the whole run: no slides, two slides with the same file stem, or a model that does not load. |
| `RunResult` | `slide`, `status` (`done`, `skipped` or `failed`), `slide_id`, `tiles`, `objects`, `wall_s`, `outputs`, `timings` (seconds per stage), `peak_memory` (MB), `error`. |

## One slide

```text
open_slide -> build_grid --> DataLoader readers --> runtime.predict --> StitchThread --> writers
 (patchology)  (tissue,       (num_workers          (this thread)       (own thread,      (GeoJSON,
               cores)          processes)                               queue of 8)       Parquet, TIFF)
```

1. **Grid.** The slide opens through patchology; `build_grid` keeps tissue tiles at the
   card's tile size and mpp, with the config's overlap (default: the card's
   `recommended_overlap`). ICC conversion is on when both the config and the card ask for
   it. A slide without a resolution raises `SlideResolutionError`.
2. **Stream** (`scheduler.stream`). Three stages, each bounded so memory stays flat
   whatever their relative speed: reader processes fill the DataLoader prefetch queue
   (`num_workers * prefetch_factor` batches), the runtime runs on the calling thread, and
   the stitcher runs behind a queue of `STITCH_QUEUE` batches. On any error the stitcher
   thread and the reader processes are stopped before the error is raised.
3. **Write.** `RunInfo` and `ObjectTable` (`io/objects.py`) feed the writers the config
   enables. Outputs go to `outputs.dir`, named after the slide's file stem.
4. **Mark.** `<stem>.done.json` is written last: settings hash, slide id and the files
   written.

## Resume

With `resume: true`, `run_batch` skips a slide when its marker exists, its
`settings_hash` matches and every file it lists still exists. The hash
(`RunConfig.settings_hash`) covers the model card and every setting that changes
outputs: tiling, stitching, outputs, class colours, autocast. It leaves out the slide
list, the loader, the device and the batch size. Change a stitch setting or swap the
model version behind an alias and the slide runs again; add slides or more workers and
finished ones stay skipped. An interrupted slide has no marker, so it runs again.

## Files per run

Besides each slide's outputs, `run_batch` writes into `outputs.dir`:

| File | Contents |
|---|---|
| `batch-<UTC time>.parquet` | One row per slide: `slide`, `status`, `slide_id`, `tiles`, `objects`, `wall_s`, `<stage>_s` timings, `peak_rss_mb`, `error`, `settings_hash`. |
| `run-<UTC time>.log` | Every pathoinfer and patchology log record of the run, DEBUG and up. |

## Logging

One INFO line per slide, as `key=value` pairs:

```text
slide=slide_name-01.svs slide_id=684c… tiles=3044 tiles_per_s=984 objects=7 wall_s=3.8
grid_s=0.70 start_s=2.88 read_s=0.08 infer_s=0.06 stitch_wait_s=0.07 stitch_finalize_s=0.01
write_s=0.02 peak_rss_mb=1299
```

| Stage | Time spent |
|---|---|
| `grid` | Opening the slide, tissue detection, building the grid |
| `start` | Starting the reader processes |
| `read` | Waiting for the next batch of tiles |
| `infer` | `runtime.predict` |
| `stitch_wait` | Waiting for room in the stitcher's queue (the stitcher is the bottleneck) |
| `stitch_finalize` | Finishing the stitch after the last batch |
| `write` | Writing the outputs |

`tiles_per_s` is tiles over `start + read + infer + stitch_wait`. A large `read` means
more workers would help; a large `stitch_wait` points at the stitcher; `infer` near the
total means the model is the limit. Run start and end, skipped and failed slides are
logged too; `-v` adds DEBUG detail.

## Performance notes

- `scripts/perf_report.py SLIDE CONFIG` times read, inference, stitching and writing one
  at a time on up to `--max-tiles` tiles and names the slowest. The per-slide log line
  shows waiting time in an overlapped run; the report shows each stage's capacity.
- Each slide gets its own DataLoader, so reader start-up (`start`, about 0.7 s per worker
  measured on one machine) is paid per slide. For a fast model on many small slides it
  can dominate; reusing readers across slides is the fix if that matters.
- Peak memory is the process's resident set (`peak_rss_mb`) and, once CUDA is in use,
  peak allocated GPU memory (`peak_cuda_mb`).

## Testing

`tests/engine/test_pipeline.py` runs a synthetic pyramidal slide end to end with the
fixture models at 4 mpp (about 100 tiles):

- segmentation writes every output, and the GeoJSON, Parquet and TIFF read back valid;
- detection with `tiny_det` (one box per tile centre) reports exactly one object per tile;
- resume skips a finished slide, and runs it again after a setting changes, an output is
  deleted, or resume is off;
- a slide without a resolution and a missing slide fail, are recorded in the batch
  summary, and the next slide still runs;
- an error mid-slide stops every reader process and writes no marker;
- 0 and 2 reader processes give the same result;
- the CLI validates, describes and runs a config, and reports bad configs.
