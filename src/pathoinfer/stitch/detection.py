"""Detection stitcher: tile detections to slide-level objects, each reported once.

Decoding. Boxes are read as the card says (``xyxy`` or ``cxcywh``, tile ``pixel``s or
``normalised`` to the tile size), turned into ``xyxy`` and moved from tile pixels at the
model's mpp to level-0 pixels: ``level0 = (tile_origin_model + box) * downsample``.
Detections below the score threshold (the stitcher's, else the card's) are dropped.

Ownership. A detection is kept only by the tile whose core contains its box centre. Cores
partition the slide, so a detection that every overlapping tile sees whole, at the same
centre, is reported exactly once. That holds for every object whose half-size is at most
``overlap // 2`` model pixels (objects up to about ``overlap`` wide): its owning tile's core
lies ``overlap // 2`` inside the tile, so the owner sees it whole. Larger objects may be
cut by tile edges, shift their predicted centre into a neighbour's core and be reported
twice; a warning is logged when the grid's overlap is below the card's
``recommended_overlap``.

NMS (optional, ``nms_iou``). For those larger objects, detections whose centre lies within
``overlap`` model pixels of an interior core edge go through class-aware NMS against
detections from other tiles (STRtree candidate pairs, greedy by score). Detections from
the same tile are never suppressed against each other: that is the model's job. Two tiles
see an object of width ``W`` cut at their own edges, so their boxes share only the
``overlap`` strip: IoU is about ``overlap / W``, and NMS removes the duplicate only when
``nms_iou`` is below that. Objects cut by two seams (a corner) overlap less still. Raising
the overlap is the dependable fix; NMS is the fallback.

Scale. Everything is held as NumPy columns (one chunk per batch, joined once); no Python
object per detection exists until a writer asks for features.
"""

import logging
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import shapely
import torch
from numpy.typing import NDArray

from pathoinfer.registry.model_card import DetectionCard
from pathoinfer.runtime.base import Outputs
from pathoinfer.stitch.base import Feature, StitchResult
from pathoinfer.tiling.grid import SlideGrid, TileSpec

log = logging.getLogger(__name__)

Geometry = Literal["point", "box"]


@dataclass(frozen=True)
class Detections:
    """Slide-level detections as columns, level-0 pixels."""

    boxes: NDArray[np.float64]
    scores: NDArray[np.float32]
    labels: NDArray[np.int64]
    class_probs: NDArray[np.float32] | None = None

    def __len__(self) -> int:
        return len(self.scores)

    def centroids(self) -> NDArray[np.float64]:
        return (self.boxes[:, :2] + self.boxes[:, 2:]) / 2

    def geometries(self, kind: Geometry = "box") -> NDArray[np.object_]:
        """Vectorised shapely Points (box centres) or box Polygons."""
        if kind == "point":
            return np.asarray(shapely.points(self.centroids()))
        return np.asarray(shapely.box(*self.boxes.T))

    def to_features(
        self, class_names: list[str], slide_mpp: float, kind: Geometry = "box"
    ) -> list[Feature]:
        """One Feature per detection, for writers. Measurements: ``score``, ``area_um2``
        (boxes only) and ``prob_<class>`` per class when the model outputs probabilities."""
        geometries = self.geometries(kind)
        widths, heights = (self.boxes[:, 2:] - self.boxes[:, :2]).T
        areas = widths * heights * slide_mpp**2
        features = []
        for i in range(len(self)):
            measurements = {"score": float(self.scores[i])}
            if kind == "box":
                measurements["area_um2"] = float(areas[i])
            if self.class_probs is not None:
                for name, p in zip(class_names, self.class_probs[i], strict=True):
                    measurements[f"prob_{name}"] = float(p)
            label = int(self.labels[i])
            features.append(Feature(geometries[i], label, class_names[label], measurements))
        return features


@dataclass
class _Chunk:
    boxes: NDArray[np.float64]
    scores: NDArray[np.float32]
    labels: NDArray[np.int64]
    probs: NDArray[np.float32] | None
    tiles: NDArray[np.int64]
    near_edge: NDArray[np.bool_]


class DetectionStitcher:
    def __init__(
        self,
        card: DetectionCard,
        grid: SlideGrid,
        *,
        score_threshold: float | None = None,
        nms_iou: float | None = None,
    ) -> None:
        self.card, self.grid = card, grid
        self.threshold = card.output.score_threshold if score_threshold is None else score_threshold
        self.nms_iou = nms_iou
        self.margin = grid.overlap * grid.downsample
        recommended = round(card.recommended_overlap * grid.tile_size)
        if grid.overlap < recommended:
            log.warning(
                "grid overlap %d px is below the card's recommended %d px; objects wider than "
                "about the overlap may be reported twice (consider nms_iou)",
                grid.overlap,
                recommended,
            )
        self._chunks: list[_Chunk] = []

    def add(self, specs: list[TileSpec], outputs: Outputs) -> None:
        out, grid = self.card.output, self.grid
        scores = _numpy(outputs[out.scores]).astype(np.float32)
        b, d = np.nonzero(scores >= self.threshold)
        if len(b) == 0:
            return
        boxes = _numpy(outputs[out.boxes])[b, d].astype(np.float64)
        if out.box_format == "cxcywh":
            half = boxes[:, 2:] / 2
            boxes = np.concatenate([boxes[:, :2] - half, boxes[:, :2] + half], axis=1)
        if out.box_units == "normalised":
            boxes *= grid.tile_size
        tiles = np.array([[s.xm, s.ym, *_core(s), s.row * grid.n_cols + s.col] for s in specs])
        origins = tiles[b, :2]
        boxes = (boxes + np.concatenate([origins, origins], axis=1)) * grid.downsample
        centre = (boxes[:, :2] + boxes[:, 2:]) / 2
        x0, y0, x1, y1 = tiles[b, 2:6].T
        owned = (
            (centre[:, 0] >= x0) & (centre[:, 0] < x1) & (centre[:, 1] >= y0) & (centre[:, 1] < y1)
        )
        interior = np.array([_interior_edges(s, grid) for s in specs])[b]
        gaps = np.stack(
            [centre[:, 0] - x0, centre[:, 1] - y0, x1 - centre[:, 0], y1 - centre[:, 1]], axis=1
        )
        near_edge = (np.where(interior, gaps, np.inf) < self.margin).any(axis=1)
        probs = None
        if out.class_probs is not None:
            probs = _numpy(outputs[out.class_probs])[b, d].astype(np.float32)[owned]
        self._chunks.append(
            _Chunk(
                boxes=boxes[owned],
                scores=scores[b, d][owned],
                labels=_numpy(outputs[out.labels])[b, d].astype(np.int64)[owned],
                probs=probs,
                tiles=tiles[b, 6].astype(np.int64)[owned],
                near_edge=near_edge[owned],
            )
        )

    def finalize(self) -> StitchResult:
        detections = self._join()
        if self.nms_iou is not None and len(detections.scores):
            keep = np.ones(len(detections.scores), bool)
            candidates = np.flatnonzero(detections.near_edge)
            suppressed = _nms(
                detections.boxes[candidates],
                detections.scores[candidates],
                detections.labels[candidates],
                detections.tiles[candidates],
                self.nms_iou,
            )
            keep[candidates[suppressed]] = False
            detections = _Chunk(
                detections.boxes[keep],
                detections.scores[keep],
                detections.labels[keep],
                None if detections.probs is None else detections.probs[keep],
                detections.tiles[keep],
                detections.near_edge[keep],
            )
        log.info("detection: %d objects", len(detections.scores))
        return StitchResult(
            features=[],
            detections=Detections(
                detections.boxes, detections.scores, detections.labels, detections.probs
            ),
        )

    def _join(self) -> _Chunk:
        if not self._chunks:
            n = self.card.output
            empty_probs = (
                None if n.class_probs is None else np.empty((0, len(self.card.classes)), np.float32)
            )
            return _Chunk(
                np.empty((0, 4)),
                np.empty(0, np.float32),
                np.empty(0, np.int64),
                empty_probs,
                np.empty(0, np.int64),
                np.empty(0, bool),
            )
        c = self._chunks
        probs = None if c[0].probs is None else np.concatenate([x.probs for x in c])
        return _Chunk(
            np.concatenate([x.boxes for x in c]),
            np.concatenate([x.scores for x in c]),
            np.concatenate([x.labels for x in c]),
            probs,
            np.concatenate([x.tiles for x in c]),
            np.concatenate([x.near_edge for x in c]),
        )


def _numpy(tensor: torch.Tensor) -> NDArray[Any]:
    return tensor.detach().cpu().numpy()


def _core(spec: TileSpec) -> tuple[int, int, int, int]:
    return spec.core.x0, spec.core.y0, spec.core.x1, spec.core.y1


def _interior_edges(spec: TileSpec, grid: SlideGrid) -> tuple[bool, bool, bool, bool]:
    """Which core edges (left, top, right, bottom) face a neighbour, not the slide edge."""
    return spec.col > 0, spec.row > 0, spec.col < grid.n_cols - 1, spec.row < grid.n_rows - 1


def _nms(
    boxes: NDArray[np.float64],
    scores: NDArray[np.float32],
    labels: NDArray[np.int64],
    tiles: NDArray[np.int64],
    iou_threshold: float,
) -> NDArray[np.int64]:
    """Indices suppressed by greedy class-aware NMS between detections of different tiles."""
    if len(scores) < 2:
        return np.empty(0, np.int64)
    tree = shapely.STRtree(shapely.box(*boxes.T))
    i, j = tree.query(shapely.box(*boxes.T))
    pair = (i < j) & (labels[i] == labels[j]) & (tiles[i] != tiles[j])
    i, j = i[pair], j[pair]
    lo = np.maximum(boxes[i, :2], boxes[j, :2])
    hi = np.minimum(boxes[i, 2:], boxes[j, 2:])
    inter = np.clip(hi - lo, 0, None).prod(axis=1)
    area = (boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
    iou = inter / (area[i] + area[j] - inter)
    i, j = i[iou > iou_threshold], j[iou > iou_threshold]
    # Each pair as (stronger, weaker); ties go to the lower index, as a stable sort would.
    stronger_i = (scores[i] > scores[j]) | ((scores[i] == scores[j]) & (i < j))
    strong, weak = np.where(stronger_i, i, j), np.where(stronger_i, j, i)
    order = np.argsort(strong, kind="stable")
    strong, weak = strong[order], weak[order]
    starts = np.searchsorted(strong, np.arange(len(scores)))
    ends = np.searchsorted(strong, np.arange(len(scores)), side="right")
    suppressed = np.zeros(len(scores), bool)
    for k in np.argsort(-scores, kind="stable"):
        if not suppressed[k] and starts[k] < ends[k]:
            suppressed[weak[starts[k] : ends[k]]] = True
    return np.flatnonzero(suppressed)
