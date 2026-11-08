"""Process slides: tissue and grid -> DataLoader -> runtime -> stitcher thread -> writers.

``run_batch`` loads the model once and runs every slide of a config. A slide that fails
is logged and recorded; the batch carries on. Each finished slide gets a
``<stem>.done.json`` marker naming its outputs and the run's settings hash, so a resumed
run skips it while the settings and outputs are unchanged.
"""

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl
from loguru import logger

from patchology import open_slide
from pathoinfer._logging import add_file_sink, log_call
from pathoinfer.config import ClassRemap, RunConfig
from pathoinfer.engine.scheduler import (
    STITCH_QUEUE,
    GpuSampler,
    Progress,
    Timings,
    cpu_seconds,
    peak_memory_mb,
    stream,
)
from pathoinfer.io.objects import ObjectTable, RunInfo
from pathoinfer.io.writers.geojson import write_geojson
from pathoinfer.io.writers.overlay import write_overlay
from pathoinfer.io.writers.parquet import write_parquet, write_summary
from pathoinfer.io.writers.tiff import write_tiff
from pathoinfer.registry.loader import load_model
from pathoinfer.registry.model_card import DetectionCard, ModelCard, SegmentationCard
from pathoinfer.runtime.base import Runtime, resolve_device
from pathoinfer.runtime.factory import make_runtime
from pathoinfer.stitch.base import Stitcher, StitchThread
from pathoinfer.stitch.detection import DetectionStitcher
from pathoinfer.stitch.segmentation import SegmentationStitcher
from pathoinfer.tiling.dataset import make_loader
from pathoinfer.tiling.grid import SlideGrid, build_grid

Status = Literal["done", "skipped", "failed"]
# The stages tiles flow through; tiles/s is tiles over their total time.
STREAM_STAGES = ("start", "read", "infer", "stitch_wait")


@dataclass(frozen=True)
class Model:
    """A loaded model, shared by every slide of a run."""

    card: ModelCard
    model_dir: Path
    runtime: Runtime
    settings_hash: str
    precision: str  # what the model runs in: the card's, or the autocast dtype


@dataclass(frozen=True)
class RunResult:
    slide: Path
    status: Status
    slide_id: str | None = None
    tiles: int = 0
    objects: int = 0
    wall_s: float = 0.0
    outputs: tuple[Path, ...] = ()
    timings: dict[str, float] = field(default_factory=dict)
    peak_memory: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@log_call
def load(cfg: RunConfig) -> Model:
    """Load and warm up ``cfg``'s model. Raises ValueError if its task is not the
    config's."""
    card, model_dir = load_model(cfg.model)
    if card.task != cfg.task:
        raise ValueError(f"config task is {cfg.task}, but {cfg.model} is a {card.task} model")
    if card.task == "classification":
        raise NotImplementedError("classification runs land in v0.2")
    logger.debug("model card: {}", card.model_dump_json())
    runtime = make_runtime(card, model_dir, cfg.runtime)
    runtime.warmup()
    autocast = card.framework == "torch" and not cfg.runtime.triton_url and cfg.runtime.autocast
    precision = autocast or card.precision
    return Model(card, model_dir, runtime, cfg.settings_hash(card.model_dump_json()), precision)


def output_paths(cfg: RunConfig, slide: Path) -> dict[str, Path]:
    """The files a slide's run writes, by kind (GeoJSON split per class adds more)."""
    out, stem = cfg.outputs, slide.stem
    paths = {}
    if out.geojson:
        paths["geojson"] = out.dir / f"{stem}.geojson{'.gz' if out.gzip else ''}"
    if out.parquet:
        paths["objects"] = out.dir / f"{stem}.objects.parquet"
        paths["summary"] = out.dir / f"{stem}.summary.{out.summary_format}"
    if out.tiff and cfg.task == "segmentation":
        paths["labels"] = out.dir / f"{stem}.labels.tiff"
    if out.overlay and cfg.task == "segmentation":
        paths["overlay"] = out.dir / f"{stem}.overlay.png"
    return paths


def finished_summary(
    name: str,
    wall: float,
    tiles: int,
    tiles_per_s: float,
    objects: int,
    timings: Timings,
    cpu: float,
    peak: dict[str, float],
) -> str:
    """A slide's run for people: time per stage (share of the wall time), the main
    process's CPU (average cores in use) and peak memory. Reader processes are not
    measured."""
    stages = sorted(timings.seconds.items(), key=lambda kv: -kv[1])
    time_line = " | ".join(
        f"{k.replace('_', ' ')} {v:.1f} s ({100 * v / max(wall, 1e-9):.0f}%)" for k, v in stages
    )
    cores = cpu / max(wall, 1e-9)

    def gb(mb: float) -> str:
        return f"{mb / 1024:.1f} GB"

    memory = f"RAM peak {gb(peak['rss_mb'])} (main process)"
    if "cuda_mb" in peak:
        memory += f"; GPU {gb(peak['cuda_mb'])} peak by torch"
    gpu = ""
    if "gpu_util_mean" in peak:
        gpu = (
            f"\n  GPU     busy {peak['gpu_util_mean']:.0f}% on average "
            f"(max {peak['gpu_util_max']:.0f}%) during inference; "
            f"{gb(peak['gpu_used_mb'])} peak in use on the device"
        )
    return (
        f"finished {name} in {wall:.1f} s: {tiles} tiles at {tiles_per_s:.0f} tiles/s, "
        f"{objects} objects\n"
        f"  time    {time_line}\n"
        f"  CPU     {cores:.1f} cores on average ({cpu:.0f} s; readers not counted)\n"
        f"  memory  {memory}{gpu}"
    )


def marker_path(cfg: RunConfig, slide: Path) -> Path:
    return cfg.outputs.dir / f"{slide.stem}.done.json"


def is_done(cfg: RunConfig, slide: Path, settings_hash: str) -> bool:
    """True if a run with the same settings finished ``slide`` and its outputs exist."""
    try:
        marker = json.loads(marker_path(cfg, slide).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return marker.get("settings_hash") == settings_hash and all(
        Path(p).is_file() for p in marker.get("outputs", [])
    )


@log_call
def run_slide(
    cfg: RunConfig, path: Path, model: Model | None = None, on_progress: Progress | None = None
) -> RunResult:
    """Run one slide end to end and write its outputs; raises on failure. ``on_progress``
    is called after each batch with (tiles done, tiles in the slide)."""
    model = model or load(cfg)
    card = model.card
    start = time.perf_counter()
    cpu_start = cpu_seconds()
    timings = Timings()
    with timings.stage("grid"), open_slide(path, cfg.tiling.backend) as slide:
        overlap = cfg.tiling.overlap
        overlap = card.recommended_overlap if overlap is None else overlap
        grid = build_grid(
            slide,
            card.input.tile_size,
            card.input.mpp,
            round(overlap * card.input.tile_size),
            min_tissue_fraction=cfg.tiling.min_tissue_fraction,
            tissue=cfg.tiling.tissue,
            artifacts=cfg.tiling.artifacts,
            icc=cfg.tiling.icc and card.input.icc,
        )
    logger.debug(
        "{}: {} tiles in a {} x {} grid", path.name, len(grid.tiles), grid.n_rows, grid.n_cols
    )
    output_mpp = segmentation_output_mpp(cfg, grid) if cfg.task == "segmentation" else None
    loader = make_loader(
        grid,
        cfg.runtime.batch_size,
        cfg.loader.num_workers,
        prefetch_factor=cfg.loader.prefetch_factor,
        persistent_workers=False,
    )
    device = resolve_device(cfg.runtime.device)
    on_gpu = device.type == "cuda" and cfg.runtime.triton_url is None  # Triton's GPU is remote
    gpu = GpuSampler(device.index or 0)
    try:
        thread = StitchThread(make_stitcher(cfg, card, grid, output_mpp), STITCH_QUEUE)
        if on_gpu:
            with gpu:
                result, tiles = stream(loader, model.runtime, thread, timings, on_progress)
        else:
            result, tiles = stream(loader, model.runtime, thread, timings, on_progress)
    finally:
        loader.dataset.close()  # type: ignore[attr-defined]  # TileDataset; open with 0 workers

    remap = (
        cfg.segmentation.class_remap(card)
        if isinstance(card, SegmentationCard)
        else ClassRemap({}, {})
    )
    run = RunInfo.create(
        card,
        grid,
        cfg.model,
        output_mpp,
        cfg.class_colors,
        exclude=remap.targets,
        rename=remap.names,
        precision=model.precision,
    )
    objects = ObjectTable.from_result(result, run, cfg.detection.geometry)
    paths = output_paths(cfg, path)
    cfg.outputs.dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with timings.stage("write"):
        if "geojson" in paths:
            written += write_geojson(paths["geojson"], objects, run, cfg.outputs.per_class)
        if "objects" in paths:
            written.append(write_parquet(paths["objects"], objects, run, cfg.outputs.geoparquet))
            written.append(write_summary(paths["summary"], objects, run))
        if "labels" in paths:
            written.append(write_tiff(paths["labels"], result.rasters["labels"], run))
        if "overlay" in paths:
            with open_slide(path, cfg.tiling.backend) as slide:
                written.append(
                    write_overlay(paths["overlay"], slide, objects, run, cfg.outputs.overlay_mpp)
                )
    marker = {
        "settings_hash": model.settings_hash,
        "precision": model.precision,
        "slide": str(path),
        "slide_id": grid.slide_id,
        "outputs": [str(p) for p in written],
        "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    marker_path(cfg, path).write_text(json.dumps(marker, indent=2) + "\n")

    wall = time.perf_counter() - start
    peak = peak_memory_mb() | gpu.result
    cpu = cpu_seconds() - cpu_start
    tiles_per_s = tiles / max(sum(timings.seconds.get(k, 0.0) for k in STREAM_STAGES), 1e-9)
    stages = " ".join(f"{k}_s={v:.2f}" for k, v in timings.seconds.items())
    memory = " ".join(f"peak_{k}={v:.0f}" for k, v in peak.items())
    logger.info(
        "slide={} slide_id={} tiles={} tiles_per_s={:.0f} objects={} wall_s={:.1f} {} "
        "cpu_s={:.0f} {}",
        path.name,
        grid.slide_id,
        tiles,
        tiles_per_s,
        len(objects),
        wall,
        stages,
        cpu,
        memory,
    )
    logger.info(
        "{}",
        finished_summary(path.name, wall, tiles, tiles_per_s, len(objects), timings, cpu, peak),
    )
    return RunResult(
        path,
        "done",
        grid.slide_id,
        tiles,
        len(objects),
        wall,
        tuple(written),
        dict(timings.seconds),
        peak,
    )


@log_call
def run_batch(cfg: RunConfig) -> list[RunResult]:
    """Run every slide of ``cfg``, skipping finished ones when ``cfg.resume``. Writes a
    ``batch-<time>.parquet`` summary and a ``run-<time>.log`` into the output folder.

    Raises:
        ValueError: no slides, two slides with the same file stem, or a task mismatch.
    """
    slides = cfg.slide_paths()
    if not slides:
        raise ValueError(f"no slides match {cfg.slides!r}")
    stems = Counter(p.stem for p in slides)
    clashes = sorted(s for s, n in stems.items() if n > 1)
    if clashes:
        raise ValueError(f"slides share output names {clashes}; give them distinct file names")
    cfg.outputs.dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    sink = add_file_sink(cfg.outputs.dir / f"run-{stamp}.log")
    start = time.perf_counter()
    try:
        model = load(cfg)
        logger.info(
            "run start: {} slides, model {} {} ({}), precision {}, settings {}",
            len(slides),
            model.card.name,
            model.card.version,
            cfg.model,
            model.precision,
            model.settings_hash,
        )
        results = [_run_one(cfg, path, model) for path in slides]
        summary = cfg.outputs.dir / f"batch-{stamp}.parquet"
        _batch_frame(results, model.settings_hash, model.precision).write_parquet(summary)
        counts = Counter(r.status for r in results)
        logger.info(
            "run end: {} done, {} skipped, {} failed in {:.1f} s; summary {}",
            counts["done"],
            counts["skipped"],
            counts["failed"],
            time.perf_counter() - start,
            summary,
        )
        return results
    finally:
        logger.remove(sink)


def _run_one(cfg: RunConfig, path: Path, model: Model) -> RunResult:
    if cfg.resume and is_done(cfg, path, model.settings_hash):
        logger.info("slide={} skipped: finished with the same settings", path.name)
        return RunResult(path, "skipped")
    try:
        return run_slide(cfg, path, model)
    except Exception as e:  # noqa: BLE001  recorded in the batch summary, the run carries on
        logger.exception("slide={} failed", path.name)
        return RunResult(path, "failed", error=f"{type(e).__name__}: {e}")


def _batch_frame(results: list[RunResult], settings_hash: str, precision: str) -> pl.DataFrame:
    stages = sorted({k for r in results for k in r.timings})
    rows = [
        {
            "slide": str(r.slide),
            "status": r.status,
            "slide_id": r.slide_id,
            "tiles": r.tiles,
            "objects": r.objects,
            "wall_s": r.wall_s,
            **{f"{k}_s": r.timings.get(k, 0.0) for k in stages},
            "peak_rss_mb": r.peak_memory.get("rss_mb"),
            "error": r.error,
            "settings_hash": settings_hash,
            "precision": precision,
        }
        for r in results
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def segmentation_output_mpp(cfg: RunConfig, grid: SlideGrid) -> float:
    return cfg.segmentation.output_mpp or max(2.0, grid.mpp)


def make_stitcher(
    cfg: RunConfig, card: ModelCard, grid: SlideGrid, output_mpp: float | None
) -> Stitcher:
    if isinstance(card, SegmentationCard):
        assert output_mpp is not None
        return SegmentationStitcher(
            card,
            grid,
            output_mpp=output_mpp,
            simplify_px=cfg.segmentation.simplify_px,
            smooth=cfg.segmentation.smooth,
            min_area_um2=cfg.segmentation.min_area_um2,
            fill_holes_um2=cfg.segmentation.fill_holes_um2,
            connectivity=cfg.segmentation.connectivity,
            keep_labels=cfg.outputs.tiff,
            class_targets=cfg.segmentation.class_remap(card).targets,
        )
    if not isinstance(card, DetectionCard):
        raise NotImplementedError(f"no stitcher for {card.task} yet")
    return DetectionStitcher(
        card,
        grid,
        score_threshold=cfg.detection.score_threshold,
        nms_iou=cfg.detection.nms_iou,
    )
