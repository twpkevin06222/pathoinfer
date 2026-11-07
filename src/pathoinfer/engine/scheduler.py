"""Move one slide's tiles from the readers through the runtime into the stitcher.

Three stages, each bounded so memory stays flat whatever their relative speed: reader
processes fill the DataLoader's prefetch queue (``num_workers * prefetch_factor``
batches), the runtime runs on this thread, and the stitcher runs on its own thread behind
a queue of ``STITCH_QUEUE`` batches. ``stream`` times each stage and, on any error,
stops the reader processes and the stitcher thread before raising.
"""

import os
import resource
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Self

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
    """Peak resident memory of this process and, on CUDA, peak GPU memory allocated by
    torch (TensorRT and ONNX Runtime allocate outside it; ``GpuSampler`` sees all). Reader
    processes are not counted: they are the forkserver's children, not this process's."""
    peaks = {"rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    if torch.cuda.is_initialized():  # type: ignore[no-untyped-call]  # never start CUDA here
        peaks["cuda_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return peaks


class GpuSampler:
    """Samples one GPU's utilisation and memory in use through NVML every ``interval_s``
    on a thread, while the ``with`` block runs. ``result`` is empty when NVML is not
    available (no NVIDIA driver), so CPU machines run unchanged.

    ``result``: ``gpu_util_mean`` and ``gpu_util_max`` (% of time a kernel ran, NVML's
    definition, not how full the GPU was) and ``gpu_used_mb`` (peak memory in use on the
    device: every process's, TensorRT's and ONNX Runtime's included)."""

    def __init__(self, cuda_index: int, interval_s: float = 0.5) -> None:
        self.cuda_index, self.interval_s = cuda_index, interval_s
        self.result: dict[str, float] = {}
        self._stop = threading.Event()
        self._util: list[int] = []
        self._used = 0
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        import pynvml

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(_nvml_index(self.cuda_index))
        except pynvml.NVMLError:
            return self
        self._thread = threading.Thread(target=self._sample, args=(handle,), daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        if self._thread is None:
            return
        import pynvml

        self._stop.set()
        self._thread.join()
        pynvml.nvmlShutdown()
        if self._util:
            self.result = {
                "gpu_util_mean": sum(self._util) / len(self._util),
                "gpu_util_max": float(max(self._util)),
                "gpu_used_mb": self._used / 2**20,
            }

    def _sample(self, handle: object) -> None:
        import pynvml

        while True:
            self._util.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            self._used = max(self._used, pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            if self._stop.wait(self.interval_s):
                return


def _nvml_index(cuda_index: int) -> int:
    """NVML numbers GPUs as the system does; CUDA renumbers them after
    ``CUDA_VISIBLE_DEVICES``. shortcut: assumes the default PCI ordering, use PCI bus ids
    if machines set CUDA_DEVICE_ORDER otherwise."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if cuda_index < len(visible) and visible[cuda_index].strip().isdigit():
        return int(visible[cuda_index])
    return cuda_index


def cpu_seconds() -> float:
    """CPU time (user + system) of this process so far, all its threads."""
    own = resource.getrusage(resource.RUSAGE_SELF)
    return own.ru_utime + own.ru_stime


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
