"""Masks to polygons, shared by the segmentation and heatmap stitchers.

Every function works in mask pixels: pixel ``(col, row)`` covers ``[col, col+1] x
[row, row+1]``. ``to_feature`` maps a region to level 0 at the end.

1. ``regions``: one ``Piece`` per 8-connected region of a mask, with holes.
   ``cv2.findContours(RETR_CCOMP, CHAIN_APPROX_SIMPLE)`` traces outer boundaries and holes
   through pixel centres; each contour is widened by half a pixel one segment at a time
   (GEOS mis-buffers a contour that runs back over itself), so polygons follow pixel
   edges and their areas equal pixel counts, except where two parts of a region touch
   only at a pixel corner: the contour runs diagonally through that corner, which adds a
   diamond of half a pixel per contact. A region's holes are subtracted in one
   difference (one at a time is quadratic in the hole count).
2. ``merge``: union same-class pieces that touch (regions cut by band boundaries).
3. ``fill_holes``: fill holes under a size unless another class's region lies inside,
   dropping same-class islands in the filled holes.
4. ``to_feature``: drop collinear vertices, simplify (Douglas-Peucker, topology kept),
   smooth (``chaikin``), make valid, scale to level 0, round to 0.1 px, drop regions under
   ``min_area_um2``; measurements ``area_um2``, ``perimeter_um``, ``mean_probability``.

``vectorise`` runs all four on a whole mask.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import MultiPolygon, Point, Polygon

from pathoinfer.stitch.base import Feature


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
    geometry = polygonal(shapely.make_valid(chaikin(geometry, smooth)))
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
) -> list[Feature]:
    """One Feature per 8-connected region of a whole ``mask``; ``values`` (same shape)
    gives each region's ``mean_probability``. See ``to_feature`` for the transform."""
    px_um2 = (level0_per_px * slide_mpp) ** 2
    pieces = fill_holes(regions(mask, values, class_id, 0), fill_holes_um2 / px_um2)
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
    mask: NDArray[np.bool_], probs: NDArray[np.float32], class_id: int, top: int
) -> list[Piece]:
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
        pieces.append(Piece(geometry, class_id, float(sums[component]), int(sizes[component])))
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


def merge(pieces: list[Piece]) -> list[Piece]:
    """Union every group of same-class pieces that touch across band boundaries."""
    merged = []
    for class_id in sorted({p.class_id for p in pieces}):
        group = [p for p in pieces if p.class_id == class_id]
        tree = shapely.STRtree([p.geometry for p in group])
        left, right = tree.query([p.geometry for p in group], predicate="intersects")
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
