"""Move one slide's tiles from the readers through the runtime into the stitcher.

Three stages, each bounded so memory stays flat whatever their relative speed: reader
processes fill the DataLoader's prefetch queue (``num_workers * prefetch_factor``
batches), the runtime runs on this thread, and the stitcher runs on its own thread behind
a queue of ``STITCH_QUEUE`` batches. ``stream`` times each stage and, on any error,
stops the reader processes and the stitcher thread before raising.
"""

import resource
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from pathoinfer.runtime.base import Runtime
from pathoinfer.stitch.base import StitchResult, StitchThread
from pathoinfer.tiling.dataset import Item

STITCH_QUEUE = 8
# Called after each batch with (tiles done, tiles in the slide).
Progress = Callable[[int, int], None]


@dataclass
class Timings:
    """Seconds per stage. ``start`` is reader start-up, ``read`` time spent waiting for
    tiles, ``stitch_wait`` time spent waiting for the stitcher's queue to have room."""

    seconds: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start


def peak_memory_mb() -> dict[str, float]:
    """Peak resident memory of this process and, on CUDA, peak allocated GPU memory."""
    peaks = {"rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if torch.cuda.is_initialized():  # type: ignore[no-untyped-call]  # never start CUDA here
        peaks["cuda_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return peaks


def stream(
    loader: DataLoader[Item],
    runtime: Runtime,
    thread: StitchThread,
    timings: Timings,
    on_progress: Progress | None = None,
) -> tuple[StitchResult, int]:
    """Run every batch of ``loader`` through ``runtime`` into ``thread``; returns the
    stitched result and the number of tiles."""
    with timings.stage("start"):  # starts the reader processes
        batches = iter(loader)
    total = len(loader.dataset.grid.tiles)  # type: ignore[attr-defined]  # TileDataset
    tiles = 0
    try:
        while True:
            with timings.stage("read"):
                batch = next(batches, None)
            if batch is None:
                break
            images, specs = batch
            with timings.stage("infer"):
                outputs = runtime.predict(images)
            with timings.stage("stitch_wait"):
                thread.put(specs, outputs)
            tiles += len(specs)
            if on_progress is not None:
                on_progress(tiles, total)
        with timings.stage("stitch_finalize"):
            return thread.finalize(), tiles
    finally:
        thread.close()
        # Stop reader processes now rather than at garbage collection, also on errors.
        shutdown = getattr(batches, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()
