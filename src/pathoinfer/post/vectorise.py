"""Masks to polygons, shared by the segmentation and heatmap stitchers.

Every function works in mask pixels: pixel ``(col, row)`` covers ``[col, col+1] x
[row, row+1]``. ``to_feature`` maps a region to level 0 at the end.

1. ``regions``: one ``Piece`` per connected region of a mask, with holes: the exact union
   of its pixels, so areas equal pixel counts and holes keep their corners. Each row's
   runs of a region become rectangles (stacked over rows where a run repeats) and are
   unioned. ``connectivity`` 8 (default) keeps parts that touch only at a pixel corner
   in one region, as polygons meeting at that point; 4 makes them separate regions.
2. ``merge``: union same-class pieces that touch (regions cut by band boundaries), by
   the same connectivity.
3. ``fill_holes``: fill holes under a size unless another class's region lies inside,
   dropping same-class islands in the filled holes.
4. ``to_feature``: drop collinear vertices, simplify (Douglas-Peucker, topology kept),
   smooth (``chaikin`` on edges split to at most ``SMOOTH_SEGMENT_PX``, so each corner
   loses about a pixel whatever the edge lengths, then simplify at
   ``SMOOTH_TOLERANCE_PX`` to drop the near-collinear points), make valid, scale to level 0, round to 0.1 px, drop regions under
   ``min_area_um2``; measurements ``area_um2``, ``perimeter_um``, ``mean_probability``.

``vectorise`` runs all four on a whole mask.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Polygon

from pathoinfer.stitch.base import Feature

Connectivity = Literal[4, 8]

# Longest edge Chaikin smoothing sees, in mask pixels: each round cuts at most a quarter
# of it (1 px) from a corner, whatever the polygon's edge lengths.
SMOOTH_SEGMENT_PX = 4.0
# Douglas-Peucker tolerance after smoothing, in mask pixels: far below what shows.
SMOOTH_TOLERANCE_PX = 0.1


@dataclass
class Piece:
    """One class region in mask pixels (one band of it, while stitching in bands)."""

    geometry: Any
    class_id: int
    prob_sum: float
    pixels: int


def to_feature(
    piece: Piece,
    class_name: str,
    *,
    level0_per_px: float,
    slide_mpp: float,
    origin: tuple[float, float] = (0.0, 0.0),
    simplify_px: float = 1.5,
    smooth: int = 2,
    min_area_um2: float = 0.0,
) -> Feature | None:
    """``piece`` as a level-0 Feature: mask pixel ``(col, row)`` maps to level-0
    ``origin + level0_per_px * (col, row)``. None if nothing polygonal is left or the
    region is under ``min_area_um2``."""
    # Collinear vertices left by the pixel-edge unions would make Douglas-Peucker pick
    # between near-tied points; drop them so it simplifies from the true corners.
    geometry = shapely.simplify(piece.geometry, 0)
    geometry = shapely.simplify(geometry, simplify_px, preserve_topology=True)
    if smooth:
        # Chaikin cuts a quarter of each edge: split long edges so cuts stay pixel-sized.
        geometry = chaikin(shapely.segmentize(geometry, SMOOTH_SEGMENT_PX), smooth)
        # Drop the near-collinear points smoothing leaves along straight runs.
        geometry = shapely.simplify(geometry, SMOOTH_TOLERANCE_PX, preserve_topology=True)
    geometry = polygonal(shapely.make_valid(geometry))
    if geometry is None:
        return None
    offset = np.asarray(origin)
    geometry = shapely.transform(geometry, lambda xy: xy * level0_per_px + offset)
    geometry = polygonal(shapely.set_precision(geometry, 0.1))
    if geometry is None or geometry.area * slide_mpp**2 < min_area_um2:
        return None
    return Feature(
        geometry=geometry,
        class_id=piece.class_id,
        class_name=class_name,
        measurements={
            "area_um2": float(geometry.area * slide_mpp**2),
            "perimeter_um": float(geometry.length * slide_mpp),
            "mean_probability": piece.prob_sum / piece.pixels,
        },
    )


def vectorise(
    mask: NDArray[np.bool_],
    values: NDArray[np.float32],
    class_id: int,
    class_name: str,
    *,
    level0_per_px: float,
    slide_mpp: float,
    origin: tuple[float, float] = (0.0, 0.0),
    simplify_px: float = 1.5,
    smooth: int = 2,
    min_area_um2: float = 0.0,
    fill_holes_um2: float = 0.0,
    connectivity: Connectivity = 8,
) -> list[Feature]:
    """One Feature per connected region of a whole ``mask``; ``values`` (same shape)
    gives each region's ``mean_probability``. See ``to_feature`` for the transform."""
    px_um2 = (level0_per_px * slide_mpp) ** 2
    pieces = regions(mask, values, class_id, 0, connectivity)
    pieces = fill_holes(pieces, fill_holes_um2 / px_um2)
    features = [
        to_feature(
            p,
            class_name,
            level0_per_px=level0_per_px,
            slide_mpp=slide_mpp,
            origin=origin,
            simplify_px=simplify_px,
            smooth=smooth,
            min_area_um2=min_area_um2,
        )
        for p in pieces
        if p.geometry.area * px_um2 >= min_area_um2
    ]
    return [f for f in features if f is not None]


def regions(
    mask: NDArray[np.bool_],
    probs: NDArray[np.float32],
    class_id: int,
    top: int,
    connectivity: Connectivity = 8,
) -> list[Piece]:
    """One piece per ``connectivity``-connected region of ``mask`` (4 or 8), with holes,
    offset by ``top`` rows: the exact union of the region's pixels. With 8, parts that
    touch only at a corner are one piece whose polygons meet at that point."""
    if not mask.any():
        return []
    count, labelled = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    labels = np.asarray(labelled, dtype=np.int64)
    sums = np.bincount(labels.ravel(), weights=probs.ravel(), minlength=count)
    sizes = np.bincount(labels.ravel(), minlength=count)
    label, x0, y0, x1, y1 = _runs(labels)
    boxes = shapely.box(x0, y0 + top, x1, y1 + top)
    bounds = np.flatnonzero(np.diff(label)) + 1
    pieces = []
    for start, stop in zip(np.r_[0, bounds], np.r_[bounds, len(label)], strict=True):
        geometry = boxes[start] if stop - start == 1 else shapely.union_all(boxes[start:stop])
        component = label[start]
        pieces.append(Piece(geometry, class_id, float(sums[component]), int(sizes[component])))
    return pieces


def _runs(labels: NDArray[np.int64]) -> tuple[NDArray[np.int64], ...]:
    """Every labelled region as rectangles ``(label, x0, y0, x1, y1)``, sorted by label:
    each row's runs of one label, stacked over rows where a run repeats unchanged."""
    padded = np.pad(labels, ((0, 0), (1, 1)))
    change = padded[:, 1:] != padded[:, :-1]  # column k: between pixels k - 1 and k
    rows, x0 = np.nonzero(change & (padded[:, 1:] != 0))
    _, x1 = np.nonzero(change & (padded[:, :-1] != 0))
    label = padded[rows, x0 + 1]
    order = np.lexsort((rows, x1, x0, label))
    label, x0, x1, rows = label[order], x0[order], x1[order], rows[order]
    same = (np.diff(label) == 0) & (np.diff(x0) == 0) & (np.diff(x1) == 0) & (np.diff(rows) == 1)
    starts = np.r_[0, np.flatnonzero(~same) + 1]
    ends = np.r_[starts[1:], len(label)] - 1
    return label[starts], x0[starts], rows[starts], x1[starts], rows[ends] + 1


def merge(pieces: list[Piece], connectivity: Connectivity = 8) -> list[Piece]:
    """Union every group of same-class pieces that touch across band boundaries: along an
    edge, or (``connectivity`` 8) also at a corner."""
    merged = []
    for class_id in sorted({p.class_id for p in pieces}):
        group = [p for p in pieces if p.class_id == class_id]
        geometries = np.array([p.geometry for p in group])
        left, right = shapely.STRtree(geometries).query(geometries, predicate="intersects")
        if connectivity == 4:  # join along a shared edge, not at a corner point
            touching = shapely.intersection(geometries[left], geometries[right])
            keep = (shapely.length(touching) > 0) | (left == right)
            left, right = left[keep], right[keep]
        for indices in _connected(len(group), zip(left.tolist(), right.tolist(), strict=True)):
            members = [group[i] for i in indices]
            merged.append(
                Piece(
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


def fill_holes(pieces: list[Piece], max_px: float) -> list[Piece]:
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
        Piece(filled.get(i, p.geometry), p.class_id, p.prob_sum, p.pixels)
        for i, p in enumerate(pieces)
        if i not in dropped
    ]


def chaikin(geometry: Any, rounds: int) -> Any:
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


def polygonal(geometry: Any) -> Any:
    """The Polygon or MultiPolygon made of ``geometry``'s polygon parts, or None."""
    parts = [
        p
        for p in shapely.get_parts(shapely.get_parts(geometry))
        if isinstance(p, Polygon) and not p.is_empty
    ]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else MultiPolygon(parts)
