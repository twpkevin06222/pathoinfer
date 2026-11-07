# Changelog

All notable changes to pathoinfer. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor release may break the
config or output formats; the entry says so.

## [Unreleased]

### Added

- `segmentation.background_classes` counts classes as background, and
  `segmentation.class_map` renames and merges classes: each key is an output name, its
  value a model class (renamed) or a list of them (merged), e.g.
  `{tumor: invasive tumor, stroma: [inflamed stroma, tumor-associated stroma]}`; classes
  not named keep theirs. Probabilities move before each pixel's class is chosen, so the
  label map, polygons, overlay and summary all show the result. `class_colors` takes the
  new names. `pathoinfer validate` checks the names against the model.
- Each slide ends with a readable summary in the log: time per stage with its share of
  the run, the main process's CPU (average cores used) and peak RAM, and on a local GPU
  its utilisation during inference (mean and max) and peak memory in use on the device,
  sampled through NVML every 0.5 s (TensorRT and ONNX Runtime memory included). Reader
  processes are not measured. The `slide=...` line gains `cpu_s`, `peak_gpu_util_mean`,
  `peak_gpu_util_max` and `peak_gpu_used_mb`. New core dependency: `nvidia-ml-py`
  (machines without an NVIDIA driver skip the GPU line).
- Segmentation runs write `<slide>.overlay.png`: the whole slide at `outputs.overlay_mpp`
  (default 6) with every class filled at 40% opacity, outlined, and a legend
  (`io.writers.overlay`). On by default; `outputs.overlay: false` turns it off.
- `outputs.summary_format: json` writes the per-class summary as `<slide>.summary.json`
  for every task: the run metadata once and `classes` mapping each class name to
  `class_id`, `count`, `mean_score` and `total_area_um2`. The default stays Parquet.
  Either new option changes the settings hash, so finished slides run again once.
- `pathoinfer convert fp16|ptq|trt|parity`:
  - FP16 conversion with ONNX Runtime's converter; I/O dtypes and the FP32 op block list
    are configurable.
  - INT8/FP8 static PTQ with NVIDIA ModelOpt (TensorRT targets, kept layers in FP16) or
    ONNX Runtime (INT8, kept layers in FP32), calibrated on tiles read through patchology
    from a slide list (entropy, minmax or percentile). First/last layers, LayerNorm,
    Softmax and attention are kept in high precision by default.
  - TensorRT plans built as strongly typed networks with a batch profile, cached by
    model, GPU, TensorRT version, precision and profile.
  - A parity gate: both models run through the real stitchers on mini-slides (Dice, F1
    and count difference, AUC per class). A pass writes the variant with
    `provenance.parity` and can register it in MLflow under an alias; a fail names the
    worst classes and layers and writes nothing.
- TensorRT plan runtime: `framework: tensorrt` model folders run in `pathoinfer run`.
- Serving:
  - `TritonRuntime` (gRPC) implements the `Runtime` protocol; `runtime.triton_url` runs a
    config's model on Triton with the engine unchanged.
  - `pathoinfer serve triton-repo`: a Triton model repository (`config.pbtxt` built as
    Triton's own protobuf, weights, card) from ONNX folders and TensorRT plans, with
    dynamic batching from a config.
  - `pathoinfer serve api`: a FastAPI slide job API (`POST /jobs`, status and progress,
    outputs, cancel), each job's engine in its own process, bounded concurrency, state on
    disk.
  - `deploy/`: one Dockerfile (CPU, or CUDA with `TORCH_INDEX=cu130`), a compose stack
    (Triton 26.10, API, demo client; the demo slide and repository are baked into the
    images) and a GPU override.
- `pathoinfer.tiling.synthetic.write_pyramid`: the synthetic slide, now in the package.
- `run_slide(..., on_progress=)`: a callback after each batch with tiles done and total.
- VLM conversion for Qwen3-VL / SmolVLM fine-tuned with TRL + PEFT
  (`docs/vlm_quantisation.md`): `pathoinfer convert vlm-merge` (LoRA into a bf16
  safetensors folder), `convert vlm` (llm-compressor `fp8`, `w4a16-awq` or
  `w4a16-gptq` on the language model only, vision encoder and projector kept bf16 by a
  verified ignore list, calibrated on pathology image + question pairs) and
  `convert vlm-parity` (PathVQA accuracy and box IoU against bf16 through two vLLM
  servers; registers only within tolerance). `runtime/remote/vllm.py` gains a minimal
  OpenAI-compatible client.
- ModelCard: `VqaCard` variant (task `vqa`, framework `hf`, precision up to `int4`),
  additive at schema version 1.
- ModelCard: optional `provenance.parity` (schema stays version 1).
- VLM runtimes (`runtime/README.md`, `docs/vlm_grounding.md` for the Qwen3-VL to SAM 3
  grounding design):
  - `VllmClient` now retries connection errors, timeouts and HTTP 408/429/5xx with
    backoff, caps requests in flight (`concurrency`, across threads), asks lists in
    order (`ask_many`), sends images as data URLs or `file://` URIs (`local_media`),
    takes RGB arrays, a system prompt, an API key and a JSON schema for the answer.
  - `runtime.vlm.parse_boxes`: every box in an answer (Qwen JSON with labels, or text),
    in the card's units and format, to level-0 px through the `Crop` it was read from.
  - `runtime/local/vllm_offline.py` (`OfflineVlm`): offline batch generation with
    vLLM's Python API, the only module that imports `vllm`. New `[vllm]` extra, not in
    `[all]`: it needs its own Python 3.13 venv, which the workspace lock resolves as a
    separate fork.
  - `pathoinfer serve vllm MODEL`: the `vllm serve` command for a VLM folder, with dtype
    and quantisation from the card and folder, `max_pixels`, and sizing from
    `VllmServeConfig`.
- `stitch.heatmap.HeatmapStitcher`: tile classification to one probability vector per
  tile core (overlap-area weighted), per-class float16 rasters, one QuPath detection per
  core with `prob_<class>` measurements, optional thresholded region polygons and
  optional Gaussian smoothing. Not yet wired into `pathoinfer run`.

### Changed

- Polygons are now the exact union of each region's pixels: areas equal pixel counts
  and holes keep their corners (before, outlines came out 1-4% too large and every hole
  lost a quarter pixel per corner). About 5x faster on noisy masks. **Segmentation
  outputs change.**
- New `segmentation.connectivity` (8, default, or 4): whether parts touching only at a
  pixel corner are one region. 8 matches the old grouping without its half-pixel joint.
- Smoothing cuts about a pixel per corner whatever the edge lengths (edges are split to
  at most 4 px first); long straight edges no longer lose large corners. Smoothed
  polygons have fewer vertices.
- The mask-to-polygon code moved from `stitch/segmentation.py` to `post/vectorise.py`
  and is shared by segmentation and heatmaps.
- `[vlm-convert]` is `llmcompressor>=0.15` and `peft>=0.21.2`. llm-compressor caps
  torch at 2.14.1 and transformers at 5.21.0, which holds the workspace lock there.
- Floors lowered for the `[vllm]` fork: torch to 2.14.0 and numpy to 2.4.6 (the main
  environment still resolves torch 2.14.1 and numpy 2.5.4).
- The VLM parity gate reads boxes with `runtime.vlm.parse_boxes` (clipped to the image)
  and its clients run `concurrency` requests at once.
- Extras: `[convert]` is now NVIDIA ModelOpt (>= 0.48), `onnx` and the helpers
  ModelOpt's ONNX quantiser imports, listed without ModelOpt's own `[onnx]` pins so
  onnxruntime-gpu 1.31 stays; `llmcompressor` moved to the new `[vlm-convert]`.

- `scripts/qc_overlay.py` draws like the run's overlay (`io.writers.overlay.render`): the
  whole slide at `--mpp` (default 6), objects filled at `--alpha` (default 0.4) with
  outlines and a class legend. The thumbnail and crops are gone, and `--thumbnail`,
  `--crop`, `--downsample` and `--seed` with them.

### Fixed

- The label map TIFF is a pyramid OpenSlide, fastslide and QuPath open. Its levels were
  SubIFDs, which OpenSlide (and QuPath through it) does not see, so it read one level
  and reported no pyramid; each level is now a top-level image. Label maps are 8-bit,
  class ids with 255 where no tile reached (was int16 with -1): OpenSlide reads 16-bit
  images as zeros. Breaking: the TIFF's dtype and no-data value change. QuPath opens
  plain TIFFs with Bio-Formats, which sees one level; choose its OpenSlide reader
  (`docs/qupath.md`).
- Segmentation post-processing is about twice as fast (95 s to 49 s on a 29,000-tile
  slide at 6 mpp): band pieces merge with a coverage union, filling a hole drops its ring
  instead of a union with the whole region, and only invalid polygons are repaired.
  Outputs are unchanged.
- A simplified border could run through a junction where only other borders end, making
  a ring touch itself; repairing that took 25 s on one large stroma polygon. Borders may
  now touch only at ends they share.
- Segmentation polygons of neighbouring classes no longer overlap. Each polygon was
  simplified and smoothed on its own, so the two copies of a shared border moved apart
  (1% of the polygon area overlapped on a real slide); borders are now shaped once per
  arc between junctions (`post.vectorise.shape_borders`). Polygon outputs change
  slightly; the synthetic ellipse's area moves by 0.04%. `to_feature` is replaced by
  `to_features`, which takes all pieces at once.
- The segmentation label TIFF now shows the same clean-up as the polygons: filled holes
  take their class and regions under `min_area_um2` become background. It was the raw
  argmax, so it held objects the GeoJSON did not. Small regions are decided once, on
  their exact pixel area; a second check after smoothing, which dropped regions whose
  smoothed outline dipped under the threshold, is gone, so slightly more small polygons
  are kept. `fill_holes` returns the filled holes too, and `to_features` takes no
  `min_area_um2`.
- `fill_holes` no longer fills a hole over another class's region that leaves the hole
  through a pixel corner (with `connectivity` 8), which made classes overlap.
- `convert fp16` converts models over 2 GB (weights in a file beside the graph); it
  failed with "Unable to parse proto" in ONNX's in-memory shape inference. Shapes are now
  inferred from the file, and a large FP16 model is saved as `model.onnx` plus
  `model.onnx.data`.
- `convert trt` builds models whose weights sit in a file beside the graph (ONNX
  models over 2 GB); it failed with "Failed to import initializer". The plan cache key
  now covers those weights, so a retrained model no longer reuses the old plan.

## [0.1.0] - 2026-11-06

First release: segmentation and detection inference on whole slide images, from a model
folder to GeoJSON, Parquet and TIFF.

### Added

- **Model contract.** ModelCard v1 for segmentation, detection and classification
  models, with a published JSON Schema (`registry/schema/model_card.v1.json`). Models
  load from a local folder or ONNX file, an MLflow registry (`models:/`, `[mlflow]`) or
  the Hugging Face Hub (`hf://`, `[hf]`). ONNX files with only legacy `pathotrain`
  metadata load with a deprecation warning.
- **CLI.** `pathoinfer run`, `validate`, `info model`, `info slide`, `card create` and
  `card validate`, with `--log-level` and `-v`.
- **Run config.** One YAML per run: tiling, runtime, reader processes, stitching options,
  outputs, slides (path, glob or `.txt`/`.csv`/`.parquet` list) and resume. Examples in
  `examples/`.
- **Tiling** through patchology: tissue detection, a grid with tile cores that partition
  the slide, parallel reader processes. Slides without a resolution are refused.
- **Runtimes.** ONNX Runtime (CPU, CUDA, TensorRT execution provider when available) and
  PyTorch (`torch.export` and TorchScript, bf16/fp16 autocast), sharing one
  preprocessing step driven by the ModelCard.
- **Segmentation stitching.** Gaussian-weighted blending of overlapping tiles, streamed
  in bands; polygons with holes on pixel edges, merged across bands; simplification,
  Chaikin smoothing (`smooth`), hole filling (`fill_holes_um2`) and small-object removal
  (`min_area_um2`). Optional label-map raster.
- **Detection stitching.** Each detection is kept by the one tile whose core holds its
  centre, so objects up to about the overlap are reported once; optional cross-tile NMS;
  box or point output with class probabilities.
- **Outputs.** QuPath GeoJSON (`.geojson[.gz]`, optionally one file per class), Parquet
  objects and per-class summary tables (WKB geometry, optional GeoParquet through DuckDB
  with a pixel CRS), pyramidal TIFF label maps. Every output records the run metadata.
- **Engine.** Batch runs that load the model once, record failed slides and carry on,
  resume finished slides by settings hash, and log one line per slide with stage
  timings and peak memory; a batch summary Parquet and a run log per batch.
- **Analysis.** DuckDB queries over many slides' Parquet files returning Polars frames:
  `class_density`, `score_histogram`.
- **Logging** with loguru through patchology's shared handler; `PATHOINFER_LOG_LEVEL`,
  `set_log_level`, `add_file_sink`.
- **Tools.** `scripts/qc_overlay.py` (alignment check), `scripts/perf_report.py`
  (per-stage throughput and bottleneck), `scripts/bench_read.py`,
  `scripts/bench_infer.py`.
- **Docs.** README, `docs/model_contract.md`, `docs/outputs.md`, `docs/qupath.md`, and a
  README per module.
- **CI.** Clean-environment install (patchology and pathoinfer only, CPU PyTorch), ruff,
  mypy, CPU tests and a wheel build.

### Not yet

- Tile classification runs (heatmaps): v0.2.
- Model conversion (FP16, INT8/FP8, TensorRT plans): v0.2.
- Serving and remote runtimes (Triton, vLLM): v0.3 and later.

[0.1.0]: https://github.com/twpkevin06222/pathoinfer/releases/tag/v0.1.0
