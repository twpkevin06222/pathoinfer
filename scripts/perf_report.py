"""Measure each stage of a run on one slide, one stage at a time, and name the bottleneck::

    uv run python scripts/perf_report.py slide_name-01.svs examples/segmentation.yaml

In a real run reading, inference and stitching overlap (reader processes, this thread,
the stitcher thread), so a run goes as fast as its slowest stage and the per-slide log
shows waiting time, not capacity. This script measures capacity: it reads up to
``--max-tiles`` tiles with the config's readers and keeps them in memory, then times
inference on those batches, then the stitcher on those outputs (adding and finalising),
then the writers. Each stage reports tiles/s; the lowest is the bottleneck. Reader
start-up and grid building are reported separately: they are paid once per slide.

One run; numbers vary with page cache, CPU load and GPU clocks, so repeat before acting
on a small difference.
"""

import argparse
import os
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch
from loguru import logger

from patchology import open_slide
from pathoinfer.config import load_config
from pathoinfer.engine.pipeline import load, make_stitcher, segmentation_output_mpp
from pathoinfer.io.objects import ObjectTable, RunInfo
from pathoinfer.io.writers.geojson import write_geojson
from pathoinfer.io.writers.parquet import write_parquet, write_summary
from pathoinfer.tiling.dataset import make_loader
from pathoinfer.tiling.grid import build_grid


def sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():  # type: ignore[no-untyped-call]
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("slide", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--max-tiles", type=int, default=4096, help="tiles to time (kept in RAM)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = load(cfg)
    card = model.card
    logger.info(
        "cpus {}, load {:.1f}, device {}, batch {}, readers {}",
        os.cpu_count(),
        os.getloadavg()[0],
        cfg.runtime.device,
        cfg.runtime.batch_size,
        cfg.loader.num_workers,
    )

    start = time.perf_counter()
    with open_slide(args.slide, cfg.tiling.backend) as slide:
        overlap = card.recommended_overlap if cfg.tiling.overlap is None else cfg.tiling.overlap
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
    grid_s = time.perf_counter() - start
    total = len(grid.tiles)
    grid = replace(grid, tiles=grid.tiles[: args.max_tiles])
    n = len(grid.tiles)

    loader = make_loader(
        grid,
        cfg.runtime.batch_size,
        cfg.loader.num_workers,
        prefetch_factor=cfg.loader.prefetch_factor,
        persistent_workers=False,
    )
    start = time.perf_counter()
    batches_iter = iter(loader)
    start_s = time.perf_counter() - start
    start = time.perf_counter()
    batches = list(batches_iter)
    read_s = time.perf_counter() - start
    loader.dataset.close()  # type: ignore[attr-defined]

    start = time.perf_counter()
    outputs = []
    for images, _ in batches:
        out = model.runtime.predict(images)
        outputs.append({k: v.cpu() for k, v in out.items()})
    sync()
    infer_s = time.perf_counter() - start

    output_mpp = segmentation_output_mpp(cfg, grid) if cfg.task == "segmentation" else None
    stitcher = make_stitcher(cfg, card, grid, output_mpp)
    start = time.perf_counter()
    for (_, specs), out in zip(batches, outputs, strict=True):
        stitcher.add(specs, out)
    add_s = time.perf_counter() - start
    start = time.perf_counter()
    result = stitcher.finalize()
    finalize_s = time.perf_counter() - start

    run = RunInfo.create(card, grid, cfg.model, output_mpp, cfg.class_colors)
    objects = ObjectTable.from_result(result, run, cfg.detection.geometry)
    with tempfile.TemporaryDirectory() as tmp:
        start = time.perf_counter()
        write_geojson(Path(tmp) / "x.geojson.gz", objects, run)
        write_parquet(Path(tmp) / "x.objects.parquet", objects, run)
        write_summary(Path(tmp) / "x.summary.parquet", objects, run)
        write_s = time.perf_counter() - start

    seen = sum(len(specs) for _, specs in batches)
    if seen != n:
        raise SystemExit(f"read {seen} tiles, expected {n}")
    stages = {
        "read": read_s,
        "inference": infer_s,
        "stitch": add_s + finalize_s,
        "write": write_s,
    }
    rates = {k: n / max(v, 1e-9) for k, v in stages.items()}
    logger.info(
        "{}: {} of {} tissue tiles timed, {} objects; grid {:.2f} s, reader start-up {:.2f} s",
        args.slide.name,
        n,
        total,
        len(objects),
        grid_s,
        start_s,
    )
    logger.info("{:<10} {:>9} {:>10}", "stage", "seconds", "tiles/s")
    for name, seconds in stages.items():
        logger.info("{:<10} {:>9.2f} {:>10.0f}", name, seconds, rates[name])
    logger.info("stitch = add {:.2f} s + finalize {:.2f} s", add_s, finalize_s)
    overlapped = {k: rates[k] for k in ("read", "inference", "stitch")}
    slowest = min(overlapped, key=overlapped.__getitem__)
    ranked = sorted(overlapped.values())
    logger.info(
        "bottleneck: {} ({:.0f} tiles/s, {:.1f}x below the next stage); a full run of {} "
        "tiles needs about {:.0f} s plus {:.1f} s start-up and grid",
        slowest,
        ranked[0],
        ranked[1] / ranked[0],
        total,
        total / ranked[0],
        start_s + grid_s,
    )


if __name__ == "__main__":
    main()
