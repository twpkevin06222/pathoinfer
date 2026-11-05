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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import shapely
import torch
import torch.nn.functional as F
from loguru import logger
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Point, Polygon

from pathoinfer.registry.model_card import SegmentationCard
from pathoinfer.runtime.base import Outputs
from pathoinfer.stitch.base import Feature, Raster, StitchResult
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


@dataclass
class _Piece:
    """One class region of one band, in output pixels."""

    geometry: Any
    class_id: int
    prob_sum: float
    pixels: int


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
        self._done: list[_Piece] = []
        self._seams: list[_Piece] = []
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
        pieces = _fill_holes(self._done + _merge(self._seams), self.fill_holes_px)
        pieces = [p for p in pieces if p.geometry.area >= self.min_area_px]
        features = [f for p in pieces if (f := self._feature(p)) is not None]
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
            for piece in _regions(labels == c, probs[c], c, top):
                piece.geometry = piece.geometry.intersection(bounds)
                if piece.geometry.is_empty:
                    continue
                _, miny, _, maxy = piece.geometry.bounds
                at_top = top > 0 and miny <= top
                at_bottom = top + rows < self.height and maxy >= top + rows
                (self._seams if at_top or at_bottom else self._done).append(piece)

    def _feature(self, piece: _Piece) -> Feature | None:
        # Collinear vertices left by the pixel-edge unions would make Douglas-Peucker pick
        # between near-tied points; drop them so it simplifies from the true corners.
        geometry = shapely.simplify(piece.geometry, 0)
        geometry = shapely.simplify(geometry, self.simplify_px, preserve_topology=True)
        geometry = _polygonal(shapely.make_valid(_chaikin(geometry, self.smooth)))
        if geometry is None:
            return None
        mpp = self.grid.slide_mpp
        geometry = shapely.transform(geometry, lambda xy: xy * self.level0_per_px)
        geometry = _polygonal(shapely.set_precision(geometry, 0.1))
        if geometry is None or geometry.area * mpp**2 < self.min_area_um2:
            return None
        name = next(c.name for c in self.card.classes if c.id == piece.class_id)
        return Feature(
            geometry=geometry,
            class_id=piece.class_id,
            class_name=name,
            measurements={
                "area_um2": float(geometry.area * mpp**2),
                "perimeter_um": float(geometry.length * mpp),
                "mean_probability": piece.prob_sum / piece.pixels,
            },
        )


def _regions(
    mask: NDArray[np.bool_], probs: NDArray[np.float32], class_id: int, top: int
) -> list[_Piece]:
    """One piece per 8-connected region of ``mask``, with holes, offset by ``top`` rows."""
    if not mask.any():
        return []
    mask8 = mask.astype(np.uint8)
    count, labelled = cv2.connectedComponents(mask8, connectivity=8)
    components = np.asarray(labelled, dtype=np.int64)
    sums = np.bincount(components.ravel(), weights=probs.ravel(), minlength=count)
    sizes = np.bincount(components.ravel(), minlength=count)
    contours, hierarchy = cv2.findContours(mask8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    links = hierarchy[0]
    pieces = []
    for i, outer in enumerate(contours):
        if links[i][3] != -1:
            continue
        holes = []
        child = links[i][2]
        while child != -1:
            holes.append(_hole(contours[child], top))
            child = links[child][0]
        # One difference with all holes: one at a time is quadratic in the hole count.
        geometry = _filled(outer, top)
        if holes:
            geometry = geometry.difference(shapely.union_all(holes))
        x, y = outer[0, 0]
        component = components[y, x]
        pieces.append(_Piece(geometry, class_id, float(sums[component]), int(sizes[component])))
    return pieces


def _ring(contour: NDArray[Any], top: int) -> tuple[Any, Any]:
    """The pixel centres of ``contour`` as (interior polygon, half-pixel-wide boundary)."""
    points = contour[:, 0, :].astype(np.float64) + (0.5, top + 0.5)
    if len(points) == 1:
        edge = Point(points[0]).buffer(0.5, cap_style="square")
    else:
        # Buffered one segment at a time: a pixel-centre contour can run back over itself,
        # and GEOS mis-buffers such a line as a whole, dropping parts of the boundary.
        ring = np.vstack([points, points[:1]])
        segments = shapely.linestrings(np.stack([ring[:-1], ring[1:]], axis=1))
        edge = shapely.union_all(shapely.buffer(segments, 0.5, cap_style="square"))
    interior = shapely.make_valid(Polygon(points)) if len(points) >= 3 else Polygon()
    return interior, edge


def _filled(contour: NDArray[Any], top: int) -> Any:
    """The pixels inside and on an outer boundary."""
    interior, edge = _ring(contour, top)
    return shapely.union(interior, edge)


def _hole(contour: NDArray[Any], top: int) -> Any:
    """The background pixels enclosed by a hole boundary (traced on the foreground)."""
    interior, edge = _ring(contour, top)
    return shapely.difference(interior, edge)


def _merge(pieces: list[_Piece]) -> list[_Piece]:
    """Union every group of same-class pieces that touch across band boundaries."""
    merged = []
    for class_id in sorted({p.class_id for p in pieces}):
        group = [p for p in pieces if p.class_id == class_id]
        tree = shapely.STRtree([p.geometry for p in group])
        left, right = tree.query([p.geometry for p in group], predicate="intersects")
        for indices in _connected(len(group), zip(left.tolist(), right.tolist(), strict=True)):
            members = [group[i] for i in indices]
            merged.append(
                _Piece(
                    shapely.unary_union([m.geometry for m in members]),
                    class_id,
                    sum(m.prob_sum for m in members),
                    sum(m.pixels for m in members),
                )
            )
    return merged


def _connected(n: int, pairs: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Connected components (union-find) of ``n`` nodes joined by ``pairs``."""
    parent = list(range(n))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[root(a)] = root(b)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return list(groups.values())


def _fill_holes(pieces: list[_Piece], max_px: float) -> list[_Piece]:
    """Fill holes smaller than ``max_px`` output px² that hold no other class's region,
    and drop the same-class islands inside them."""
    if max_px <= 0 or not pieces:
        return pieces
    geometries = [p.geometry for p in pieces]
    tree = shapely.STRtree(geometries)
    dropped: set[int] = set()
    filled: dict[int, Any] = {}
    for i, piece in enumerate(pieces):
        polygons = list(getattr(piece.geometry, "geoms", [piece.geometry]))
        holes = [Polygon(ring) for poly in polygons for ring in poly.interiors]
        small = [h for h in holes if h.area < max_px]
        if not small:
            continue
        hole_ids, inside = tree.query(small, predicate="contains")
        blocked = {
            h for h, j in zip(hole_ids, inside, strict=True) if pieces[j].class_id != piece.class_id
        }
        fill = [small[h] for h in range(len(small)) if h not in blocked]
        if not fill:
            continue
        dropped |= {int(j) for h, j in zip(hole_ids, inside, strict=True) if h not in blocked}
        filled[i] = shapely.union_all([piece.geometry, *fill])
    return [
        _Piece(filled.get(i, p.geometry), p.class_id, p.prob_sum, p.pixels)
        for i, p in enumerate(pieces)
        if i not in dropped
    ]


def _chaikin(geometry: Any, rounds: int) -> Any:
    """``rounds`` of Chaikin corner cutting on every ring of a (Multi)Polygon."""
    if rounds == 0:
        return geometry

    def cut(ring: Any) -> NDArray[np.float64]:
        points = np.asarray(ring.coords)[:-1]
        for _ in range(rounds):
            following = np.roll(points, -1, axis=0)
            points = np.stack(
                [0.75 * points + 0.25 * following, 0.25 * points + 0.75 * following], axis=1
            ).reshape(-1, 2)
        return points

    polygons = [
        Polygon(cut(p.exterior), [cut(h) for h in p.interiors])
        for p in getattr(geometry, "geoms", [geometry])
        if not p.is_empty
    ]
    return polygons[0] if len(polygons) == 1 else shapely.MultiPolygon(polygons)


def _polygonal(geometry: Any) -> Any:
    """The Polygon or MultiPolygon made of ``geometry``'s polygon parts, or None."""
    parts = [
        p
        for p in shapely.get_parts(shapely.get_parts(geometry))
        if isinstance(p, Polygon) and not p.is_empty
    ]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)
