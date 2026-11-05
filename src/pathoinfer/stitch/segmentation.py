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
becomes one polygon per 8-connected region: ``cv2.findContours(RETR_CCOMP,
CHAIN_APPROX_SIMPLE)`` traces outer boundaries and holes through pixel centres, and each
ring is widened by half a pixel (mitred) so polygons follow pixel edges. Polygons that
reach a band boundary are merged with the polygons they touch in the neighbouring band
(STRtree + unary_union) once every band is done.

Clean-up, per merged region (whole regions, so band boundaries never matter): holes
under ``fill_holes_um2`` are filled unless another class's region lies inside them, and
same-class islands inside a filled hole are dropped; regions under ``min_area_um2`` are
dropped before any further work.

Post-processing, per polygon: simplify (``simplify_px`` output pixels, topology kept),
which turns pixel staircases into diagonal runs; smooth with ``smooth`` rounds of
Chaikin corner cutting (each round replaces every vertex with points 1/4 and 3/4 along
its two edges, doubling the vertices and converging to a quadratic B-spline, so outlines
are curves rather than pixel steps); make valid, keep polygonal parts, drop regions under ``min_area_um2``, scale to level 0
and round to 0.1 px. Measurements: ``area_um2``, ``perimeter_um`` and ``mean_probability``
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

from pathoinfer.post.vectorise import Piece, fill_holes, merge, regions, to_feature
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
        band_height: int | None = None,
        keep_labels: bool = False,
    ) -> None:
        if output_mpp < grid.mpp:
            raise ValueError(f"output_mpp {output_mpp} is finer than the model's {grid.mpp}")
        self.card, self.grid = card, grid
        self.simplify_px, self.min_area_um2 = simplify_px, min_area_um2
        self.smooth = smooth
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
        self.classes = [c.id for c in card.classes if c.id != card.background_id]
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
        pieces = fill_holes(self._done + merge(self._seams), self.fill_holes_px)
        pieces = [p for p in pieces if p.geometry.area >= self.min_area_px]
        names = {c.id: c.name for c in self.card.classes}
        features = [
            f
            for p in pieces
            if (
                f := to_feature(
                    p,
                    names[p.class_id],
                    level0_per_px=self.level0_per_px,
                    slide_mpp=self.grid.slide_mpp,
                    simplify_px=self.simplify_px,
                    smooth=self.smooth,
                    min_area_um2=self.min_area_um2,
                )
            )
            is not None
        ]
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
        labels = probs.argmax(0).astype(np.int16)
        labels[~covered] = -1
        if self._labels is not None:
            self._labels[top : top + rows] = labels
        bounds = shapely.box(0, top, self.width, top + rows)
        for c in self.classes:
            for piece in regions(labels == c, probs[c], c, top):
                piece.geometry = piece.geometry.intersection(bounds)
                if piece.geometry.is_empty:
                    continue
                _, miny, _, maxy = piece.geometry.bounds
                at_top = top > 0 and miny <= top
                at_bottom = top + rows < self.height and maxy >= top + rows
                (self._seams if at_top or at_bottom else self._done).append(piece)
