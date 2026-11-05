"""Stitcher protocol, its result, and the thread that runs it.

A stitcher turns tile outputs into one slide-level result in level-0 pixel coordinates.
It runs on its own CPU thread (:class:`StitchThread`), fed through a bounded queue, so
the model never waits on stitching and a slow stitcher back-pressures the reader instead
of filling memory.
"""

import logging
import queue
import threading
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Protocol, Self

import numpy as np
from numpy.typing import NDArray

from pathoinfer.runtime.base import Outputs
from pathoinfer.tiling.grid import TileSpec

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Feature:
    """One slide-level object: a shapely Polygon or MultiPolygon in level-0 pixels."""

    geometry: Any
    class_id: int
    class_name: str
    measurements: dict[str, float]


@dataclass(frozen=True)
class Raster:
    """A slide-level image. Level-0 transform: pixel ``(col, row)`` covers level-0
    ``origin + scale * (col, row)`` to ``origin + scale * (col + 1, row + 1)``."""

    array: NDArray[np.generic]
    scale: float
    origin: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class StitchResult:
    """``features`` for polygon tasks; ``detections`` (a columnar table, see
    ``stitch.detection.Detections``) for detection, turned into features by writers."""

    features: list[Feature]
    rasters: dict[str, Raster] = field(default_factory=dict)
    detections: Any = None


class Stitcher(Protocol):
    def add(self, specs: list[TileSpec], outputs: Outputs) -> None:
        """Merge one batch: ``outputs`` row ``i`` (any device) belongs to ``specs[i]``.
        Batches may arrive in any tile order."""
        ...

    def finalize(self) -> StitchResult:
        """Called once, after the last batch."""
        ...


_DONE = object()


class StitchThread:
    """Run ``stitcher`` on a background thread fed by a queue of ``maxsize`` batches.

    ``put`` blocks while the queue is full. An error in the stitcher is raised from the
    next ``put`` or from ``finalize``; the thread keeps draining the queue meanwhile, so
    the producer never deadlocks.
    """

    def __init__(self, stitcher: Stitcher, maxsize: int = 8) -> None:
        self.stitcher = stitcher
        self._queue: queue.Queue[Any] = queue.Queue(maxsize)
        self._error: BaseException | None = None
        self._result: StitchResult | None = None
        self._thread = threading.Thread(target=self._work, name="stitcher", daemon=True)
        self._thread.start()

    def put(self, specs: list[TileSpec], outputs: Outputs) -> None:
        self._raise()
        self._queue.put((specs, outputs))

    def finalize(self) -> StitchResult:
        self._queue.put(_DONE)
        self._thread.join()
        self._raise()
        assert self._result is not None
        return self._result

    def _work(self) -> None:
        finalize = True
        while (item := self._queue.get()) is not _DONE and item is not None:
            if self._error is None:
                try:
                    self.stitcher.add(*item)
                except Exception as e:
                    log.exception("stitcher failed")
                    self._error = e
        if item is None:
            finalize = False
        if finalize and self._error is None:
            try:
                self._result = self.stitcher.finalize()
            except Exception as e:
                log.exception("stitcher failed to finalize")
                self._error = e

    def _raise(self) -> None:
        if self._error is not None:
            raise RuntimeError("stitcher failed") from self._error

    def close(self) -> None:
        """Stop without finalizing (e.g. after an error upstream)."""
        if self._thread.is_alive():
            self._queue.put(None)
            self._thread.join()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
