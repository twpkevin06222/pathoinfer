"""Heatmap stitcher: per-tile class probabilities to one vector per tile core.

Averaging. Each kept tile gives one probability vector (softmax of its logits, unless the
card says the model outputs probabilities). A core's vector is the mean of every kept
tile that overlaps the core, weighted by the overlap area, so a core's own tile counts
most and neighbours count for the overlap strip they share. Overlap areas factor into an
x and a y part, both computed from the grid's tile and core extents (model pixels), so the
average is a few array shifts. Cores of tiles that were not kept (no tissue) have no value.

Smoothing (optional, ``sigma_tiles``). A Gaussian over the core grid, sigma in tiles,
normalised by the smoothed valid mask so cores without a value neither count nor change.

Outputs (``finalize``):

- ``rasters["prob_<class>"]``: float16, one pixel per core, NaN without a value. Pixel
  ``(col, row)`` covers level-0 ``origin + scale * (col, row)`` with ``scale`` the grid
  stride and ``origin`` half the overlap; that is each interior core exactly. Cores on
  the slide edge reach the slide edge, so their pixel covers half an overlap less.
- ``detections``: one object per kept core (its level-0 box), ``labels`` the argmax class,
  ``scores`` its probability, ``class_probs`` every class's probability. Writers emit them
  as QuPath detections with a ``prob_<class>`` measurement each, for measurement maps.
- ``features``: for each class in ``thresholds``, region polygons where its probability
  is at least the threshold, traced on the core grid by ``post.vectorise`` (QuPath
  annotations). ``mean_probability`` is the class's mean over the region's cores. With
  ``connectivity`` 8 (default) cores touching only at a corner are one region; with 4,
  separate regions.
"""

import math
from collections.abc import Mapping

import cv2
import numpy as np
import torch
from loguru import logger
from numpy.typing import NDArray

from pathoinfer.post.vectorise import Connectivity, vectorise
from pathoinfer.registry.model_card import ClassificationCard
from pathoinfer.runtime.base import Outputs
from pathoinfer.stitch.base import Feature, Raster, StitchResult
from pathoinfer.stitch.detection import Detections
from pathoinfer.tiling.grid import SlideGrid, TileSpec


class HeatmapStitcher:
    def __init__(
        self,
        card: ClassificationCard,
        grid: SlideGrid,
        *,
        sigma_tiles: float = 0.0,
        thresholds: Mapping[str, float] | None = None,
        simplify_px: float = 1.0,
        smooth: int = 2,
        min_area_um2: float = 0.0,
        fill_holes_um2: float = 0.0,
        connectivity: Connectivity = 8,
    ) -> None:
        """``thresholds`` maps class names to the probability their regions start at.
        Region options are as in ``post.vectorise``, in core-grid pixels."""
        names = {c.name: c.id for c in card.classes}
        thresholds = dict(thresholds or {})
        unknown = set(thresholds) - set(names)
        if unknown:
            raise ValueError(f"thresholds for unknown classes {sorted(unknown)}")
        bad = {k: v for k, v in thresholds.items() if not 0 < v <= 1}
        if bad:
            raise ValueError(f"thresholds must be in (0, 1], got {bad}")
        if sigma_tiles < 0:
            raise ValueError(f"sigma_tiles must be >= 0, got {sigma_tiles}")
        self.card, self.grid = card, grid
        self.sigma_tiles = sigma_tiles
        self.thresholds = {names[k]: v for k, v in thresholds.items()}
        self.region_options = {
            "simplify_px": simplify_px,
            "smooth": smooth,
            "min_area_um2": min_area_um2,
            "fill_holes_um2": fill_holes_um2,
            "connectivity": connectivity,
        }
        n = card.output.n_classes
        self._probs = np.zeros((grid.n_rows, grid.n_cols, n), np.float32)
        self._seen = np.zeros((grid.n_rows, grid.n_cols), bool)

    def add(self, specs: list[TileSpec], outputs: Outputs) -> None:
        with torch.inference_mode():
            scores = outputs[self.card.output.tensor].float()
            if self.card.output.activation == "logits":
                scores = scores.softmax(dim=1)
            probs = scores.cpu().numpy()
        rows = np.array([s.row for s in specs])
        cols = np.array([s.col for s in specs])
        self._probs[rows, cols] = probs
        self._seen[rows, cols] = True

    def finalize(self) -> StitchResult:
        cores = self.core_probabilities()
        kept = [s for s in self.grid.tiles if self._seen[s.row, s.col]]
        if len(kept) < len(self.grid.tiles):
            logger.warning("{} grid tiles never arrived", len(self.grid.tiles) - len(kept))
        scale, origin = self.transform()
        rasters = {
            f"prob_{c.name}": Raster(cores[..., c.id].astype(np.float16), scale, origin)
            for c in self.card.classes
        }
        tile_probs = cores[[s.row for s in kept], [s.col for s in kept]]
        boxes = np.array([[s.core.x0, s.core.y0, s.core.x1, s.core.y1] for s in kept], float)
        labels = tile_probs.argmax(1) if len(kept) else np.empty(0, np.int64)
        tiles = Detections(
            boxes.reshape(-1, 4),
            tile_probs.max(1, initial=0).astype(np.float32),
            labels.astype(np.int64),
            tile_probs.astype(np.float32),
        )
        names = {c.id: c.name for c in self.card.classes}
        regions: list[Feature] = []
        for class_id, threshold in self.thresholds.items():
            values = np.nan_to_num(cores[..., class_id], nan=0.0)
            regions += vectorise(
                values >= threshold,
                values,
                class_id,
                names[class_id],
                level0_per_px=scale,
                slide_mpp=self.grid.slide_mpp,
                origin=origin,
                **self.region_options,  # type: ignore[arg-type]
            )
        logger.info("heatmap: {} tiles, {} regions", len(kept), len(regions))
        return StitchResult(features=regions, rasters=rasters, detections=tiles)

    def transform(self) -> tuple[float, tuple[float, float]]:
        """Level-0 size of one core-grid pixel and the grid's level-0 origin."""
        grid = self.grid
        offset = grid.overlap // 2 * grid.downsample
        return (grid.tile_size - grid.overlap) * grid.downsample, (offset, offset)

    def core_probabilities(self) -> NDArray[np.float32]:
        """``(n_rows, n_cols, n_classes)``: each kept core's overlap-weighted (and, with
        ``sigma_tiles``, smoothed) probabilities; NaN for cores of tiles not kept."""
        grid = self.grid
        weights_y = _overlaps(grid.n_rows, grid.tile_size, grid.overlap)
        weights_x = _overlaps(grid.n_cols, grid.tile_size, grid.overlap)
        valid = self._seen.astype(np.float32)
        total = np.zeros_like(self._probs)
        weight = np.zeros_like(valid)
        for dy, wy in weights_y.items():
            for dx, wx in weights_x.items():
                w = wy[:, None] * wx[None, :] * _shift(valid, dy, dx)
                total += w[..., None] * _shift(self._probs, dy, dx)
                weight += w
        cores = np.divide(
            total, weight[..., None], out=np.zeros_like(total), where=weight[..., None] > 0
        )
        if self.sigma_tiles > 0:
            cores = _smooth(cores, self._seen, self.sigma_tiles)
        cores[~self._seen] = np.nan
        return cores


def _overlaps(n: int, tile: int, overlap: int) -> dict[int, NDArray[np.float32]]:
    """For each offset ``d``, the length (model px) by which tile ``i + d`` overlaps the
    core of tile ``i``, along one axis of ``n`` tiles."""
    stride = tile - overlap
    i = np.arange(n)
    origin = i * stride
    core_lo = origin + np.where(i > 0, overlap // 2, 0)
    core_hi = origin + tile - np.where(i < n - 1, overlap - overlap // 2, 0)
    reach = math.ceil(tile / stride)
    out = {}
    for d in range(-reach, reach + 1):
        tile_lo, tile_hi = origin + d * stride, origin + d * stride + tile
        length = np.clip(np.minimum(core_hi, tile_hi) - np.maximum(core_lo, tile_lo), 0, None)
        length[(i + d < 0) | (i + d >= n)] = 0
        if length.any():
            out[d] = length.astype(np.float32)
    return out


def _shift(array: NDArray[np.float32], dy: int, dx: int) -> NDArray[np.float32]:
    """``out[r, c] = array[r + dy, c + dx]``, zero where that falls outside."""
    out = np.zeros_like(array)
    rows, cols = array.shape[:2]
    out[max(0, -dy) : rows - max(0, dy), max(0, -dx) : cols - max(0, dx)] = array[
        max(0, dy) : rows - max(0, -dy), max(0, dx) : cols - max(0, -dx)
    ]
    return out


def _smooth(
    cores: NDArray[np.float32], seen: NDArray[np.bool_], sigma: float
) -> NDArray[np.float32]:
    """Gaussian over the core grid, normalised by the smoothed valid mask."""
    radius = max(1, math.ceil(3 * sigma))
    kernel = cv2.getGaussianKernel(2 * radius + 1, sigma, cv2.CV_32F)
    valid = seen.astype(np.float32)

    def blur(image: NDArray[np.float32]) -> NDArray[np.float32]:
        out = cv2.sepFilter2D(image, -1, kernel, kernel, borderType=cv2.BORDER_CONSTANT)
        return np.asarray(out, dtype=np.float32)

    weight = blur(valid)
    smoothed = np.stack(
        [blur(np.ascontiguousarray(cores[..., c] * valid)) for c in range(cores.shape[2])],
        axis=-1,
    )
    normalised = np.divide(
        smoothed, weight[..., None], out=np.zeros_like(smoothed), where=weight[..., None] > 0
    )
    return np.asarray(normalised, dtype=np.float32)
