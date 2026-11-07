# pathoinfer - scripts

Developer tools for checking outputs and measuring speed. They are not part of the
installed package. Run them from `packages/pathoinfer/` with `uv run python scripts/...`.
Each script's docstring has the details; `--help` lists every option.

| Script | Question it answers |
|---|---|
| `qc_overlay.py` | What did a run find, and does it line up with the tissue? |
| `perf_report.py` | Which stage limits a run on this slide and config? |
| `bench_read.py` | How fast are tiles read, per number of reader processes? |
| `bench_infer.py` | How fast does a model run, per runtime and precision? |

Scripts that start reader processes must be run as files, not piped through stdin: the
readers start from a fresh interpreter (`forkserver`) that re-imports the script.

## qc_overlay.py

```bash
uv run python scripts/qc_overlay.py slide_name-01.svs outputs/slide_name-01.geojson.gz
```

Writes `<name>.qc.png` next to the GeoJSON (or `--out`): the whole slide read at
`--mpp`, with every object filled in its class colour at `--alpha` opacity and outlined
at full opacity, and a legend of the classes. Detections (points) are circles.

| Option | Default | Meaning |
|---|---|---|
| `--mpp` | `6.0` | Image resolution, µm per px (the slide must record its own) |
| `--alpha` | `0.4` | Fill opacity: 0 shows only outlines, 1 hides the tissue |

At 6 mpp a 40 x 30 mm slide is about 6700 x 5000 px. Outlines are drawn on pixel
edges, as the coordinates mean. To check alignment itself, use a model whose output you
can predict from the image (a threshold model works): a model that labels glass and
tissue alike outlines only the processed tiles.

## perf_report.py

```bash
uv run python scripts/perf_report.py slide_name-01.svs examples/segmentation.yaml
```

In a run, reading, inference and stitching overlap, so the per-slide log shows waiting
time. This script measures each stage's capacity instead: it reads up to `--max-tiles`
tiles (default 4096, kept in memory) with the config's readers, then times inference on
them, then the stitcher (`add` and `finalize`), then the writers. Output:

```text
stage        seconds    tiles/s
read            0.41       4970
inference       0.04      46774
stitch         67.25         30
write           0.19      10901
stitch = add 23.08 s + finalize 44.17 s
bottleneck: stitch (30 tiles/s, 163.2x below the next stage); a full run of 39382 tiles
needs about 1293 s plus 7.1 s start-up and grid
```

The bottleneck is the slowest of read, inference and stitch; writing runs once per slide
and is reported but not ranked. Grid building and reader start-up are reported separately.
The numbers come from one run: repeat it before acting on a small difference.

## bench_read.py

```bash
uv run python scripts/bench_read.py slide_name-01.svs --workers 0 2 4 8
```

Tile reads per second through `tiling.dataset` for each reader count, on the first
`--max-tiles` tissue tiles. An untimed pass warms the page cache first (`--cold` skips
it), so the numbers measure decoding and resizing, not the disk. Start-up time per reader
count is reported separately. Every run checks that it read each tile once at the right
shape and that all runs read the same pixels. Tile geometry: `--mpp`, `--tile-size`,
`--overlap`, `--batch-size`.

## bench_infer.py

```bash
uv run python scripts/bench_infer.py MODEL_DIR [MODEL_DIR ...] --batch-size 16
```

Tiles per second for each runtime a model's framework allows: ONNX on CPU and CUDA
(`--tensorrt` adds the TensorRT execution provider), PyTorch on CPU and CUDA without
autocast and with bf16 and fp16. Input is a pinned uint8 batch, as the reader delivers
it, so the time includes the copy to the GPU and preprocessing. Each configuration warms
up (`--warmup`), then runs `--iters` batches `--repeats` times; the table shows the median
and the range. Outputs are checked for shape and finite values on every batch.
`--cpu-only` skips CUDA.
