# Changelog

All notable changes to pathoinfer. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). Before 1.0, a minor release may break the
config or output formats; the entry says so.

## [Unreleased]

### Added

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
- ModelCard: optional `provenance.parity` (schema stays version 1).
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
- Extras: `[convert]` is now NVIDIA ModelOpt (>= 0.48), `onnx` and the helpers
  ModelOpt's ONNX quantiser imports, listed without ModelOpt's own `[onnx]` pins so
  onnxruntime-gpu 1.31 stays; `llmcompressor` moved to the new `[vlm-convert]`.

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
