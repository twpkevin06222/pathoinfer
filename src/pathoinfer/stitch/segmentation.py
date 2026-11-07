"""Segmentation stitcher: tile logits to slide-level class polygons.

Resolution. Everything is accumulated at ``output_mpp`` (default 2.0, never finer than
the model's mpp). With ``s = output_mpp / model_mpp`` model pixels per output pixel, a
tile of ``tile_size`` model pixels covers ``round(tile_size / s)`` output pixels and starts
at ``round(model_origin / s)``; one output pixel is ``output_mpp / slide_mpp`` level-0
pixels.

Blending. Each tile's class probabilities (softmax of the logits, unless the card says the
model already outputs probabilities) are area-resampled to the output resolution and
weighted by a centre-heavy Gaussian (nnU-Net style, sigma = tile / 8), then summed into a
probability canvas next to a weight canvas; their ratio is the blended probability.

Bands. The canvas is held as horizontal bands of ``band_height`` output rows. Each band
knows how many of the grid's tiles overlap it and is finalised when the last of them has
arrived, whatever the arrival order, so memory is a few bands, not the slide. Peak
canvas memory is logged.

Polygons. A finalised band is argmaxed to a label map. Each class but the background
becomes one polygon per connected region (``connectivity`` 8 or 4), traced exactly on
pixel edges by ``post.vectorise.regions``. Polygons that
reach a band boundary are merged with the polygons they touch in the neighbouring band
(STRtree + unary_union) once every band is done.

Clean-up, per merged region (whole regions, so band boundaries never matter): holes
under ``fill_holes_um2`` are filled unless another class's region lies inside them, and
same-class islands inside a filled hole are dropped; regions under ``min_area_um2`` are
dropped before any further work.

Post-processing, all regions together as one coverage (``post.vectorise.to_features``),
so neighbouring classes keep sharing one border line and never overlap: simplify
(``simplify_px`` output pixels), which turns pixel staircases into diagonal runs; smooth
with ``smooth`` rounds of Chaikin corner cutting per border arc (each round replaces every
vertex with points 1/4 and 3/4 along its two edges, so outlines are curves rather than
pixel steps); make valid, keep polygonal parts, scale to level 0 and round to 0.1 px.

Label map (``keep_labels``): the argmax, with the same clean-up painted in (filled holes
take their class, removed regions become the background), so it holds the same objects
as the polygons, on exact pixels. Measurements: ``area_um2``, ``perimeter_um`` and ``mean_probability``
(the blended probability of the polygon's class over its pixels).
"""

import math
from dataclasses import dataclass

import numpy as np
import shapely
import torch
import torch.nn.functional as F
from loguru import logger
from numpy.typing import NDArray

from pathoinfer.post.vectorise import (
    Connectivity,
    Piece,
    fill_holes,
    merge,
    regions,
    to_features,
)
from pathoinfer.registry.model_card import SegmentationCard
from pathoinfer.runtime.base import Outputs
from pathoinfer.stitch.base import Raster, StitchResult
from pathoinfer.tiling.grid import SlideGrid, TileSpec


def gaussian_window(size: int) -> NDArray[np.float32]:
    """``size`` x ``size`` Gaussian with sigma ``size / 8``, peak 1; zeros are raised to
    the smallest positive value so every covered pixel gets some weight (as in nnU-Net)."""
    offsets = np.arange(size) - (size - 1) / 2
    line = np.exp(-(offsets**2) / (2 * (size / 8) ** 2))
    window = np.outer(line, line)
    window /= window.max()
    positive: NDArray[np.float32] = np.maximum(window, window[window > 0].min()).astype(np.float32)
    return positive


@dataclass
class _Band:
    probs: NDArray[np.float32]
    weight: NDArray[np.float32]


class SegmentationStitcher:
    def __init__(
        self,
        card: SegmentationCard,
        grid: SlideGrid,
        *,
        output_mpp: float = 2.0,
        simplify_px: float = 1.5,
        smooth: int = 2,
        min_area_um2: float = 0.0,
        fill_holes_um2: float = 0.0,
        connectivity: Connectivity = 8,
        band_height: int | None = None,
        keep_labels: bool = False,
        class_targets: dict[int, int] | None = None,
    ) -> None:
        """``class_targets`` (``SegmentationOptions.class_remap``) moves each listed
        class's probability to another class before the argmax."""
        if output_mpp < grid.mpp:
            raise ValueError(f"output_mpp {output_mpp} is finer than the model's {grid.mpp}")
        self.card, self.grid = card, grid
        self.simplify_px, self.min_area_um2 = simplify_px, min_area_um2
        self.smooth, self.connectivity = smooth, connectivity
        px_um2 = output_mpp**2
        self.fill_holes_px, self.min_area_px = fill_holes_um2 / px_um2, min_area_um2 / px_um2
        self.scale = output_mpp / grid.mpp
        self.level0_per_px = output_mpp / grid.slide_mpp
        width, height = grid.slide_dimensions
        self.width = math.ceil(width / self.level0_per_px)
        self.height = math.ceil(height / self.level0_per_px)
        self.tile = max(1, round(grid.tile_size / self.scale))
        self.window = gaussian_window(self.tile)
        self.band_height = band_height or 2 * self.tile
        self.targets = class_targets or {}
        self.classes = [
            c.id for c in card.classes if c.id != card.background_id and c.id not in self.targets
        ]
        self._pending = np.zeros(math.ceil(self.height / self.band_height), np.int64)
        for spec in grid.tiles:
            for band in self._bands_of(spec):
                self._pending[band] += 1
        self._bands: dict[int, _Band] = {}
        self._done: list[Piece] = []
        self._seams: list[Piece] = []
        self._labels = np.full((self.height, self.width), -1, np.int16) if keep_labels else None
        self._bytes = self.peak_bytes = 0

    def add(self, specs: list[TileSpec], outputs: Outputs) -> None:
        with torch.inference_mode():
            x = outputs[self.card.output.tensor].float()
            probs = x.softmax(1) if self.card.output.activation == "logits" else x
            if probs.shape[-1] != self.tile or probs.shape[-2] != self.tile:
                probs = F.interpolate(probs, size=(self.tile, self.tile), mode="area")
            tiles = probs.cpu().numpy() * self.window
        for spec, tile in zip(specs, tiles, strict=True):
            self._accumulate(spec, tile)

    def finalize(self) -> StitchResult:
        if self._bands:
            logger.warning(
                "{} bands were missing tiles; finalising them as they are", len(self._bands)
            )
            for band in sorted(self._bands):
                self._finish_band(band)
        pieces = self._done + merge(self._seams, self.connectivity)
        pieces, holes = fill_holes(pieces, self.fill_holes_px)
        small = [p for p in pieces if p.geometry.area < self.min_area_px]
        pieces = [p for p in pieces if p.geometry.area >= self.min_area_px]
        if self._labels is not None:  # the label map shows the same clean-up as the polygons
            background = -1 if self.card.background_id is None else self.card.background_id
            for piece in holes:
                _paint(self._labels, piece.geometry, piece.class_id)
            for piece in small:
                _paint(self._labels, piece.geometry, background)
        names = {c.id: c.name for c in self.card.classes}
        features = to_features(
            pieces,
            names,
            level0_per_px=self.level0_per_px,
            slide_mpp=self.grid.slide_mpp,
            simplify_px=self.simplify_px,
            smooth=self.smooth,
        )
        logger.info(
            "segmentation: {} polygons; peak canvas {:.1f} MiB (bands of {} x {} px at {:.3g} mpp)",
            len(features),
            self.peak_bytes / 2**20,
            self.band_height,
            self.width,
            self.level0_per_px * self.grid.slide_mpp,
        )
        rasters = {}
        if self._labels is not None:
            rasters["labels"] = Raster(self._labels, self.level0_per_px)
        return StitchResult(features, rasters)

    def _origin(self, spec: TileSpec) -> tuple[int, int]:
        return round(spec.xm / self.scale), round(spec.ym / self.scale)

    def _bands_of(self, spec: TileSpec) -> range:
        _, y = self._origin(spec)
        last_row = min(y + self.tile, self.height) - 1
        if last_row < y:
            return range(0)
        return range(y // self.band_height, last_row // self.band_height + 1)

    def _accumulate(self, spec: TileSpec, tile: NDArray[np.float32]) -> None:
        x, y = self._origin(spec)
        cols = min(self.tile, self.width - x)
        for b in self._bands_of(spec):
            band = self._band(b)
            top = b * self.band_height
            r0, r1 = max(y, top), min(y + self.tile, top + band.weight.shape[0])
            band.probs[:, r0 - top : r1 - top, x : x + cols] += tile[:, r0 - y : r1 - y, :cols]
            band.weight[r0 - top : r1 - top, x : x + cols] += self.window[r0 - y : r1 - y, :cols]
            self._pending[b] -= 1
            if self._pending[b] == 0:
                self._finish_band(b)

    def _band(self, b: int) -> _Band:
        if b not in self._bands:
            rows = min(self.band_height, self.height - b * self.band_height)
            n = self.card.output.n_classes
            band = _Band(
                np.zeros((n, rows, self.width), np.float32),
                np.zeros((rows, self.width), np.float32),
            )
            self._bands[b] = band
            self._bytes += band.probs.nbytes + band.weight.nbytes
            self.peak_bytes = max(self.peak_bytes, self._bytes)
        return self._bands[b]

    def _finish_band(self, b: int) -> None:
        band = self._bands.pop(b)
        self._bytes -= band.probs.nbytes + band.weight.nbytes
        top, rows = b * self.band_height, band.weight.shape[0]
        covered = band.weight > 0
        probs = np.divide(band.probs, band.weight, out=np.zeros_like(band.probs), where=covered)
        for source, target in self.targets.items():  # dropped and merged classes
            probs[target] += probs[source]
            probs[source] = 0
        labels = probs.argmax(0).astype(np.int16)
        labels[~covered] = -1
        if self._labels is not None:
            self._labels[top : top + rows] = labels
        bounds = shapely.box(0, top, self.width, top + rows)
        for c in self.classes:
            for piece in regions(labels == c, probs[c], c, top, self.connectivity):
                piece.geometry = piece.geometry.intersection(bounds)
                if piece.geometry.is_empty:
                    continue
                _, miny, _, maxy = piece.geometry.bounds
                at_top = top > 0 and miny <= top
                at_bottom = top + rows < self.height and maxy >= top + rows
                (self._seams if at_top or at_bottom else self._done).append(piece)


def _paint(labels: NDArray[np.int16], geometry: shapely.Geometry, value: int) -> None:
    """Set the pixels of ``labels`` whose centres lie in ``geometry`` (output px, on pixel
    edges) to ``value``. Only for small regions: it tests every pixel of the bounding box."""
    x0, y0, x1, y1 = (round(b) for b in geometry.bounds)
    rows, cols = np.mgrid[y0:y1, x0:x1]
    inside = shapely.contains_xy(geometry, cols + 0.5, rows + 0.5)
    labels[y0:y1, x0:x1][inside] = value
