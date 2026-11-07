import cv2
import numpy as np
import pytest
import shapely

from pathoinfer.post.vectorise import (
    fill_holes,
    merge,
    regions,
    shape_borders,
    to_features,
    vectorise,
)


def pixels(mask: np.ndarray) -> shapely.Geometry:
    rows, cols = np.nonzero(mask)
    return shapely.union_all(shapely.box(cols, rows, cols + 1, rows + 1))


@pytest.mark.parametrize("connectivity", [4, 8])
@pytest.mark.parametrize("seed", range(5))
def test_regions_are_exactly_the_mask_pixels(seed: int, connectivity: int) -> None:
    noise = np.random.default_rng(seed).standard_normal((96, 128)).astype(np.float32)
    mask = cv2.GaussianBlur(noise, (0, 0), 1.5) > 0.1
    pieces = regions(mask, mask.astype(np.float32), 1, 0, connectivity)  # type: ignore[arg-type]
    assert all(p.geometry.is_valid for p in pieces)
    assert sum(p.geometry.area for p in pieces) == mask.sum()
    assert shapely.equals(shapely.union_all([p.geometry for p in pieces]), pixels(mask))
    count, _ = cv2.connectedComponents(mask.astype(np.uint8), connectivity=connectivity)
    assert len(pieces) == count - 1
    assert sum(p.pixels for p in pieces) == mask.sum()


@pytest.mark.parametrize("hole", [(1, 1), (3, 4), (5, 5)])
def test_holes_keep_their_corners(hole: tuple[int, int]) -> None:
    mask = np.zeros((30, 30), bool)
    mask[5:25, 5:25] = True
    mask[10 : 10 + hole[0], 10 : 10 + hole[1]] = False
    (piece,) = regions(mask, mask.astype(np.float32), 1, 0)
    (ring,) = piece.geometry.interiors
    assert shapely.Polygon(ring).area == hole[0] * hole[1]


def test_corner_contact_is_one_region_or_two_by_connectivity() -> None:
    mask = np.zeros((20, 20), bool)
    mask[5:10, 5:10] = True
    mask[10:15, 10:15] = True
    values = mask.astype(np.float32)
    (joined,) = regions(mask, values, 1, 0, connectivity=8)
    assert joined.geometry.area == 50 and joined.geometry.is_valid  # no diamond at the joint
    assert sorted(p.geometry.area for p in regions(mask, values, 1, 0, connectivity=4)) == [25, 25]


@pytest.mark.parametrize(("connectivity", "expected"), [(8, 1), (4, 2)])
def test_band_seams_merge_by_the_same_connectivity(connectivity: int, expected: int) -> None:
    """Two squares that touch only at a corner, cut by a band boundary at row 10."""
    mask = np.zeros((20, 20), bool)
    mask[5:10, 5:10] = True
    mask[10:15, 10:15] = True
    values = mask.astype(np.float32)
    top = regions(mask[:10], values[:10], 1, 0, connectivity)  # type: ignore[arg-type]
    bottom = regions(mask[10:], values[10:], 1, 10, connectivity)  # type: ignore[arg-type]
    assert len(merge(top + bottom, connectivity)) == expected  # type: ignore[arg-type]


def test_band_offset_moves_rows() -> None:
    mask = np.zeros((4, 4), bool)
    mask[1, 2] = True
    (piece,) = regions(mask, mask.astype(np.float32), 1, top=100)
    assert piece.geometry.bounds == (2, 101, 3, 102)


@pytest.mark.parametrize("side", [8, 40, 200])
def test_smoothing_rounds_corners_by_about_a_pixel_whatever_the_size(side: int) -> None:
    mask = np.zeros((side + 20, side + 20), bool)
    mask[10 : 10 + side, 10 : 10 + side] = True
    (smooth,) = vectorise(mask, mask.astype(np.float32), 1, "t", level0_per_px=1, slide_mpp=1)
    square = shapely.box(10, 10, 10 + side, 10 + side)
    assert square.buffer(1e-6).contains(smooth.geometry)
    assert smooth.geometry.contains(square.buffer(-1.0, join_style="mitre"))
    assert square.area - smooth.geometry.area < 4 * 1.5  # about a pixel per corner


def test_vectorise_maps_pixels_to_level0() -> None:
    mask = np.zeros((10, 10), bool)
    mask[2:4, 3:7] = True
    (f,) = vectorise(
        mask,
        mask.astype(np.float32),
        1,
        "t",
        level0_per_px=8,
        slide_mpp=0.5,
        origin=(100, 50),
        simplify_px=0,
        smooth=0,
    )
    assert f.geometry.bounds == (100 + 3 * 8, 50 + 2 * 8, 100 + 7 * 8, 50 + 4 * 8)
    assert f.measurements["area_um2"] == pytest.approx(8 * 64 * 0.25)


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("smooth", [0, 2])
def test_neighbouring_classes_share_their_borders(seed: int, smooth: int) -> None:
    """Regions of a noisy label map, simplified and smoothed: no two overlap, and together
    they cover what the pixels cover, so every shared border is one line."""
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.random((80, 80)).astype(np.float32), (0, 0), 3)
    labels = np.digitize(noise, np.quantile(noise, [0.3, 0.6]))
    values = np.ones(labels.shape, np.float32)
    pieces = [p for c in (0, 1, 2) for p in regions(labels == c, values, c + 1, 0)]
    features = to_features(
        pieces, {1: "a", 2: "b", 3: "c"}, level0_per_px=1, slide_mpp=1, smooth=smooth
    )
    geometries = np.array([f.geometry for f in features])
    left, right = shapely.STRtree(geometries).query(geometries, predicate="intersects")
    pairs = left < right
    overlap = shapely.area(shapely.intersection(geometries[left[pairs]], geometries[right[pairs]]))
    assert overlap.max(initial=0) < 0.05  # rounding to 0.1 px, nothing more
    assert shapely.union_all(geometries).area == pytest.approx(80 * 80, abs=2)


def test_hole_is_kept_when_another_class_leaves_it_through_a_corner() -> None:
    """Class 2 fills a hole in class 1 and also continues outside it through a pixel
    corner: it is one 8-connected region, not contained in the hole, and must still keep
    the hole open."""
    labels = np.ones((7, 7), int)
    labels[0, :] = labels[:, 0] = 0
    labels[2:4, 2:4] = 2  # the hole
    labels[1, 1] = 2  # leaves through the corner at (2, 2)
    values = np.ones(labels.shape, np.float32)
    pieces = [p for c in (1, 2) for p in regions(labels == c, values, c, 0, 8)]
    filled, _ = fill_holes(pieces, max_px=100)
    by_class = {p.class_id: p.geometry for p in filled}
    assert by_class[1].intersection(by_class[2]).area == 0
    assert by_class[1].area == (labels == 1).sum()


def test_a_simplified_border_never_runs_through_a_junction() -> None:
    """From a real slide: simplifying class 2's border drew a segment through the corner
    where only other borders end, so its ring touched itself (and repairing the whole
    slide's stroma polygon took 25 s)."""
    labels = np.array(
        [[2, 2, 2, 2, 2, 2], [2, 7, 7, 7, 7, 2], [7, 7, 2, 0, 7, 7], [7, 2, 2, 7, 2, 2]]
    )
    values = np.ones(labels.shape, np.float32)
    pieces = [p for c in (2, 7) for p in regions(labels == c, values, c, 0)]
    shaped = shape_borders(np.array([p.geometry for p in pieces]), 1.5, 0)
    assert shapely.is_valid(shaped).all()
