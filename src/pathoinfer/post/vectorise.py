"""Masks to polygons, shared by the segmentation and heatmap stitchers.

Every function works in mask pixels: pixel ``(col, row)`` covers ``[col, col+1] x
[row, row+1]``. ``to_features`` maps regions to level 0 at the end.

1. ``regions``: one ``Piece`` per connected region of a mask, with holes: the exact union
   of its pixels, so areas equal pixel counts and holes keep their corners. Each row's
   runs of a region become rectangles (stacked over rows where a run repeats) and are
   unioned. ``connectivity`` 8 (default) keeps parts that touch only at a pixel corner
   in one region, as polygons meeting at that point; 4 makes them separate regions.
2. ``merge``: union same-class pieces that touch (regions cut by band boundaries), by
   the same connectivity.
3. ``fill_holes``: fill holes under a size unless another class's region lies inside,
   dropping same-class islands in the filled holes.
4. ``to_features``: all regions together, so a border two regions share stays one line
   and polygons never overlap or leave gaps between them (``shape_borders``). Borders are
   cut into arcs at junctions and each arc is processed once, ends fixed: simplify
   (Douglas-Peucker), smooth (``chaikin`` on edges split to at most
   ``SMOOTH_SEGMENT_PX``, so each corner loses about a pixel), simplify again at
   ``SMOOTH_TOLERANCE_PX`` to drop near-collinear points. Then make valid, scale to level
   0 and round to 0.1 px; measurements ``area_um2``,
   ``perimeter_um``, ``mean_probability``.

``vectorise`` runs all four on a whole mask. ``label_pieces`` is ``regions`` for a label map
(one piece per id, e.g. nucleus instances, so touching instances stay apart).
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise
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

GRID_PX = 0.1  # output coordinates snap to this level-0 grid (shapely precision)


@dataclass
class Piece:
    """One class region in mask pixels (one band of it, while stitching in bands)."""

    geometry: Any
    class_id: int
    prob_sum: float
    pixels: int
    extra: dict[str, float] = field(default_factory=dict)  # copied to its Feature's measurements


def to_features(
    pieces: list[Piece],
    names: dict[int, str],
    *,
    level0_per_px: float,
    slide_mpp: float,
    origin: tuple[float, float] = (0.0, 0.0),
    simplify_px: float = 1.5,
    smooth: int = 2,
) -> list[Feature]:
    """``pieces`` (non-overlapping, on pixel edges) as level-0 Features: mask pixel
    ``(col, row)`` maps to level-0 ``origin + level0_per_px * (col, row)``. Pieces are
    simplified and smoothed as one coverage, so a border two regions share stays one line
    and polygons never overlap. Callers drop small regions before, on their exact pixel
    area, so the polygons and a label map agree on which objects exist."""
    if not pieces:
        return []
    geometries = shape_borders(np.array([p.geometry for p in pieces]), simplify_px, smooth)
    # Repair only what needs it: make_valid on every smoothed outline costs ~0.5 ms each.
    invalid = ~shapely.is_valid(geometries)
    geometries[invalid] = shapely.make_valid(geometries[invalid])
    offset = np.asarray(origin)
    features = []
    for piece, geometry in zip(pieces, geometries, strict=True):
        geometry = polygonal(geometry)
        if geometry is None:
            continue
        geometry = shapely.transform(geometry, lambda xy: xy * level0_per_px + offset)
        geometry = polygonal(shapely.set_precision(geometry, GRID_PX))
        if geometry is None:
            continue
        features.append(
            Feature(
                geometry=geometry,
                class_id=piece.class_id,
                class_name=names[piece.class_id],
                measurements={
                    "area_um2": float(geometry.area * slide_mpp**2),
                    "perimeter_um": float(geometry.length * slide_mpp),
                    "mean_probability": piece.prob_sum / piece.pixels,
                    **piece.extra,
                },
            )
        )
    return features


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
    gives each region's ``mean_probability``. See ``to_features`` for the transform."""
    px_um2 = (level0_per_px * slide_mpp) ** 2
    pieces = regions(mask, values, class_id, 0, connectivity)
    pieces, _ = fill_holes(pieces, fill_holes_um2 / px_um2)
    return to_features(
        [p for p in pieces if p.geometry.area * px_um2 >= min_area_um2],
        {class_id: class_name},
        level0_per_px=level0_per_px,
        slide_mpp=slide_mpp,
        origin=origin,
        simplify_px=simplify_px,
        smooth=smooth,
    )


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


def label_pieces(
    labels: NDArray[np.integer],
    values: NDArray[np.float32],
    class_of: NDArray[np.int64],
) -> list[Piece]:
    """One piece per non-zero id of the label map ``labels``: the exact union of its pixels
    (a MultiPolygon if they are not connected), with class ``class_of[id]`` and ``values``
    summed over it. Ids that touch stay separate pieces sharing their border, so
    ``to_features`` shapes that border once for both."""
    ids: NDArray[np.int64] = np.asarray(labels, dtype=np.int64)
    if not ids.any():
        return []
    count = int(ids.max()) + 1
    sums = np.bincount(ids.ravel(), weights=values.ravel(), minlength=count)
    sizes = np.bincount(ids.ravel(), minlength=count)
    label, x0, y0, x1, y1 = _runs(ids)
    boxes = shapely.box(x0, y0, x1, y1)
    bounds = np.flatnonzero(np.diff(label)) + 1
    pieces = []
    for start, stop in zip(np.r_[0, bounds], np.r_[bounds, len(label)], strict=True):
        geometry = boxes[start] if stop - start == 1 else shapely.union_all(boxes[start:stop])
        i = int(label[start])
        pieces.append(Piece(geometry, int(class_of[i]), float(sums[i]), int(sizes[i])))
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
                    _join([m.geometry for m in members]),
                    class_id,
                    sum(m.prob_sum for m in members),
                    sum(m.pixels for m in members),
                )
            )
    return merged


def _join(geometries: list[Any]) -> Any:
    """Union of non-overlapping pieces on pixel edges. Splitting edges at every pixel makes
    shared edges match vertex for vertex, so the fast coverage union applies; collinear
    points are dropped after."""
    if len(geometries) == 1:
        return geometries[0]
    union = shapely.coverage_union_all(shapely.segmentize(np.array(geometries), 1.0))
    return shapely.simplify(union, 0, preserve_topology=False)  # topology-safe at 0; far faster


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


def fill_holes(pieces: list[Piece], max_px: float) -> tuple[list[Piece], list[Piece]]:
    """Fill holes smaller than ``max_px`` output px² that hold no other class's region,
    and drop the same-class islands inside them.
    Also returns the filled holes, one piece each with the class that now covers it, so a
    label map can be painted to match."""
    if max_px <= 0 or not pieces:
        return pieces, []
    geometries = [p.geometry for p in pieces]
    tree = shapely.STRtree(geometries)
    parts, owner = shapely.get_parts(geometries, return_index=True)
    part_tree = shapely.STRtree(parts)
    dropped: set[int] = set()
    filled: dict[int, Any] = {}
    holes_filled: list[Piece] = []
    for i, piece in enumerate(pieces):
        polygons = list(getattr(piece.geometry, "geoms", [piece.geometry]))
        rings = [(k, ring) for k, poly in enumerate(polygons) for ring in poly.interiors]
        small_ids = [n for n, (_, ring) in enumerate(rings) if Polygon(ring).area < max_px]
        small = [Polygon(rings[n][1]) for n in small_ids]
        if not small:
            continue
        hole_ids, inside = tree.query(small, predicate="contains")
        # Another class's region blocks the hole if any polygon of it lies inside: an
        # 8-connected region can leave the hole through a pixel corner, so the region as a
        # whole is not contained, but its polygon inside the hole is.
        part_holes, part_ids = part_tree.query(small, predicate="contains")
        blocked = {
            h
            for h, j in zip(part_holes, owner[part_ids], strict=True)
            if pieces[j].class_id != piece.class_id
        }
        fill = [small[h] for h in range(len(small)) if h not in blocked]
        if not fill:
            continue
        dropped |= {int(j) for h, j in zip(hole_ids, inside, strict=True) if h not in blocked}
        # Filling a hole is dropping its ring: exact, and far cheaper than a union with
        # a large region (islands inside are same-class and dropped above).
        gone = {small_ids[h] for h in range(len(small)) if h not in blocked}
        kept_rings = [
            [ring for n, (k, ring) in enumerate(rings) if k == idx and n not in gone]
            for idx in range(len(polygons))
        ]
        parts = [Polygon(poly.exterior, r) for poly, r in zip(polygons, kept_rings, strict=True)]
        filled[i] = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        holes_filled += [Piece(h, piece.class_id, 0.0, 0) for h in fill]
    kept = [
        Piece(filled.get(i, p.geometry), p.class_id, p.prob_sum, p.pixels)
        for i, p in enumerate(pieces)
        if i not in dropped
    ]
    return kept, holes_filled


def shape_borders(geometries: NDArray[Any], simplify_px: float, rounds: int) -> NDArray[Any]:
    """Simplify and smooth non-overlapping polygons on pixel edges so that a border two
    of them share stays one line, and no two borders cross.

    Edges are split at every pixel, so a shared border has the same vertices on both
    sides. Borders are cut into arcs at junctions (vertices on other than two distinct
    edges: three regions meeting, or regions touching at a corner), and each distinct arc
    is processed once, ends fixed, in a canonical direction (from its smaller end; a loop
    from its smallest vertex): Douglas-Peucker at ``simplify_px``, ``rounds`` of
    ``chaikin`` on edges split to at most ``SMOOTH_SEGMENT_PX``, Douglas-Peucker at
    ``SMOOTH_TOLERANCE_PX``. Arcs a pixel apart (both sides of a thin strand) can then
    cross; an arc that touches another anywhere but at their shared ends steps back to
    smoothing only, then to its exact pixel line, until nothing crosses."""
    parts, part_owner = shapely.get_parts(shapely.segmentize(geometries, 1.0), return_index=True)
    rings, ring_part = shapely.get_rings(parts, return_index=True)
    coords, ring_of = shapely.get_coordinates(rings, return_index=True)
    _, vertex = np.unique(coords, axis=0, return_inverse=True)
    vertex = vertex.ravel()
    same_ring = ring_of[:-1] == ring_of[1:]
    a, b = vertex[:-1][same_ring], vertex[1:][same_ring]
    edges = np.unique(np.sort(np.stack([a, b], axis=1), axis=1), axis=0)
    junction = np.bincount(edges.ravel(), minlength=vertex.max() + 1) != 2

    arc_ids: dict[bytes, int] = {}
    arcs: list[NDArray[np.float64]] = []
    loops: list[bool] = []
    ring_arcs: list[list[tuple[int, bool]]] = []  # per ring: (arc, walked backwards)

    def arc_id(points: NDArray[np.float64], loop: bool) -> tuple[int, bool]:
        canonical, flip = _canonical(points, loop)
        key = canonical.tobytes() + bytes([loop])
        if key not in arc_ids:
            arc_ids[key] = len(arcs)
            arcs.append(canonical)
            loops.append(loop)
        return arc_ids[key], flip

    bounds = np.r_[0, np.flatnonzero(~same_ring) + 1, len(coords)]
    for start, stop in pairwise(bounds):
        points, ids = coords[start : stop - 1], vertex[start : stop - 1]  # open ring
        cuts = np.flatnonzero(junction[ids])
        if len(cuts) == 0:
            ring_arcs.append([arc_id(np.vstack([points, points[:1]]), True)])
            continue
        points = np.roll(points, -cuts[0], axis=0)
        points = np.vstack([points, points[:1]])
        cuts = np.r_[cuts - cuts[0], len(points) - 1]
        ring_arcs.append([arc_id(points[i : j + 1], False) for i, j in pairwise(cuts)])

    level = np.full(len(arcs), 2)  # 2: simplify and smooth, 1: smooth, 0: pixel line
    shaped: list[NDArray[np.float64]] = list(arcs)
    redo = np.arange(len(arcs))
    while len(redo):
        for i in redo:
            tolerance = simplify_px if level[i] == 2 else 0.0
            shaped[i] = _shape_arc(arcs[i], tolerance, rounds if level[i] else 0, loops[i])
        redo = np.array(sorted(_crossing(shaped, loops)), dtype=np.int64)
        redo = redo[level[redo] > 0]
        level[redo] -= 1

    shaped_rings = []
    for ring in ring_arcs:
        pieces = [shaped[i][::-1] if flip else shaped[i] for i, flip in ring]
        shaped_rings.append(np.vstack([q[:-1] for q in pieces] + [pieces[0][:1]]))
    ring_bounds = np.r_[0, np.flatnonzero(np.diff(ring_part)) + 1, len(ring_part)]
    polygons = [Polygon(shaped_rings[i], shaped_rings[i + 1 : j]) for i, j in pairwise(ring_bounds)]
    out = np.empty(len(geometries), dtype=object)
    for members in np.split(np.arange(len(parts)), np.flatnonzero(np.diff(part_owner)) + 1):
        out[part_owner[members[0]]] = (
            polygons[members[0]]
            if len(members) == 1
            else MultiPolygon([polygons[i] for i in members])
        )
    return out


def _canonical(points: NDArray[np.float64], loop: bool) -> tuple[NDArray[np.float64], bool]:
    """An arc in its canonical direction, and whether ``points`` run against it. A loop
    (first point = last) starts at its smallest vertex and heads to the smaller neighbour;
    its direction does not matter to the ring, so it is never reported flipped."""
    if loop:
        ring = points[:-1]
        ring = np.roll(ring, -np.lexsort(ring.T[::-1])[0], axis=0)
        if tuple(ring[-1]) < tuple(ring[1]):
            ring = np.vstack([ring[:1], ring[:0:-1]])
        return np.vstack([ring, ring[:1]]), False
    flip = tuple(points[-1]) < tuple(points[0]) or (
        tuple(points[-1]) == tuple(points[0]) and tuple(points[-2]) < tuple(points[1])
    )
    return (points[::-1] if flip else points), flip


def _shape_arc(
    points: NDArray[np.float64], simplify_px: float, rounds: int, loop: bool
) -> NDArray[np.float64]:
    """Douglas-Peucker at ``simplify_px`` (0 still drops collinear points), then
    ``rounds`` of Chaikin and Douglas-Peucker at ``SMOOTH_TOLERANCE_PX``; ends fixed."""
    make = shapely.LinearRing if loop else shapely.LineString
    shaped: NDArray[np.float64] = shapely.get_coordinates(
        shapely.simplify(make(points), simplify_px)
    )
    if rounds:
        shaped = chaikin(_densify(shaped), rounds, closed=loop)
        shaped = shapely.get_coordinates(shapely.simplify(make(shaped), SMOOTH_TOLERANCE_PX))
    return shaped


def _crossing(arcs: list[NDArray[np.float64]], loops: list[bool]) -> set[int]:
    """Arcs that cross themselves, or touch another arc anywhere but at shared ends."""
    lines = np.array([shapely.LineString(arc) for arc in arcs], dtype=object)
    bad = set(np.flatnonzero(~shapely.is_simple(lines)).tolist())
    left, right = shapely.STRtree(lines).query(lines, predicate="intersects")
    pairs = left < right
    left, right = left[pairs], right[pairs]
    ends = np.array(
        [
            shapely.MultiPoint() if loop else shapely.multipoints(arc[[0, -1]])
            for arc, loop in zip(arcs, loops, strict=True)
        ]
    )
    contact = shapely.intersection(lines[left], lines[right])
    # Only an end both arcs share may touch: a segment simplified through a junction where
    # only other arcs end makes the ring touch itself.
    extra = shapely.difference(contact, shapely.intersection(ends[left], ends[right]))
    hit = ~shapely.is_empty(extra)
    return bad | set(left[hit].tolist()) | set(right[hit].tolist())


def _densify(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """``points`` (a polyline) with each edge split into equal parts of at most
    ``SMOOTH_SEGMENT_PX``. ``(a * (n - k) + b * k) / n`` gives the same points walked
    from either end."""
    a, b = points[:-1], points[1:]
    n = np.maximum(np.ceil(np.hypot(*(b - a).T) / SMOOTH_SEGMENT_PX), 1).astype(np.int64)
    edge = np.repeat(np.arange(len(a)), n)
    k = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n))[:, None]
    n = n[edge][:, None]
    return np.vstack([(a[edge] * (n - k) + b[edge] * k) / n, points[-1:]])


def chaikin(points: NDArray[np.float64], rounds: int, *, closed: bool) -> NDArray[np.float64]:
    """``rounds`` of Chaikin corner cutting on a polyline (first point = last point when
    ``closed``); an open one keeps its ends. Each round replaces every edge ``(p, q)`` with
    ``0.75 p + 0.25 q`` and ``0.25 p + 0.75 q``."""
    for _ in range(rounds):
        p, q = points[:-1], points[1:]
        cut = np.stack([0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q], axis=1).reshape(-1, 2)
        points = np.vstack([cut, cut[:1]]) if closed else np.vstack([points[:1], cut, points[-1:]])
    return points


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
