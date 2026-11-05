from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import shapely
import torch

from pathoinfer.registry.model_card import ClassificationCard, parse_card
from pathoinfer.stitch.heatmap import HeatmapStitcher
from pathoinfer.tiling.grid import SlideGrid, TileSpec

from .test_segmentation import DOWNSAMPLE, MODEL_MPP, OVERLAP, SLIDE_MPP, TILE, grid

STRIDE = TILE - OVERLAP  # 224 model px between tiles
Model = Callable[[TileSpec], float]  # tumour probability per tile


def card(activation: str = "probs") -> ClassificationCard:
    parsed = parse_card(
        {
            "name": "tiles",
            "version": "1",
            "task": "classification",
            "framework": "onnx",
            "file": "model.onnx",
            "input": {
                "tile_size": TILE,
                "mpp": MODEL_MPP,
                "normalisation": {"mean": [0.5] * 3, "std": [0.5] * 3},
            },
            "classes": [
                {"id": 0, "name": "normal", "color": [0, 160, 0]},
                {"id": 1, "name": "tumour", "color": [220, 0, 0]},
            ],
            "background_id": 0,
            "output": {"tensor": "scores", "activation": activation, "n_classes": 2},
            "recommended_overlap": OVERLAP / TILE,
        }
    )
    assert isinstance(parsed, ClassificationCard)
    return parsed


def run(model: Model, slide: SlideGrid | None = None, **options: Any) -> Any:
    slide = slide or grid()
    stitcher = HeatmapStitcher(card(), slide, **options)
    specs = list(slide.tiles)
    for start in range(0, len(specs), 7):
        chunk = specs[start : start + 7]
        p = torch.tensor([model(s) for s in chunk])
        stitcher.add(chunk, {"scores": torch.stack([1 - p, p], dim=1)})
    return stitcher, stitcher.finalize()


def checkerboard(spec: TileSpec) -> float:
    """Tumour on 3 x 3-tile blocks whose block row + column is odd."""
    return 0.9 if (spec.row // 3 + spec.col // 3) % 2 else 0.1


def brute_force(slide: SlideGrid, model: Model) -> np.ndarray:
    """Each core's tumour probability by intersecting its level-0 box with every tile."""
    out = np.full((slide.n_rows, slide.n_cols), np.nan)
    tiles = list(slide.tiles)
    tile_boxes = shapely.box(*np.array([[t.x, t.y, t.x + t.size, t.y + t.size] for t in tiles]).T)
    for spec in tiles:
        core = shapely.box(spec.core.x0, spec.core.y0, spec.core.x1, spec.core.y1)
        weights = shapely.area(shapely.intersection(tile_boxes, core))
        out[spec.row, spec.col] = np.dot(weights, [model(t) for t in tiles]) / weights.sum()
    return out


def test_checkerboard_raster_matches_the_brute_force_average() -> None:
    _, result = run(checkerboard)
    raster = result.rasters["prob_tumour"]
    assert raster.array.dtype == np.float16 and raster.array.shape == (9, 9)
    np.testing.assert_allclose(raster.array, brute_force(grid(), checkerboard), atol=1e-3)
    np.testing.assert_allclose(result.rasters["prob_normal"].array + raster.array, 1.0, atol=2e-3)
    assert raster.scale == STRIDE * DOWNSAMPLE
    assert raster.origin == (OVERLAP // 2 * DOWNSAMPLE,) * 2
    # Block interiors see only their own block.
    assert raster.array[4, 4] == pytest.approx(0.1, abs=1e-3)
    assert raster.array[1, 4] == pytest.approx(0.9, abs=1e-3)


def test_one_hot_tile_spreads_by_the_overlap_areas() -> None:
    """A core of side s = 224 model px is covered by its own tile (s * s) and by its
    neighbours' overlap strips (s * 16 per side, 16 * 16 per corner): (s + 32)^2 in all."""
    _, result = run(lambda s: 1.0 if (s.row, s.col) == (4, 4) else 0.0)
    a = result.rasters["prob_tumour"].array.astype(np.float64)
    total = (STRIDE + OVERLAP) ** 2
    assert a[4, 4] == pytest.approx(STRIDE**2 / total, abs=1e-3)
    assert a[4, 5] == pytest.approx(STRIDE * OVERLAP / 2 / total, abs=1e-3)
    assert a[3, 3] == pytest.approx((OVERLAP / 2) ** 2 / total, abs=1e-3)
    assert a[4, 6] == 0 and a[2, 4] == 0


def test_cores_without_a_tile_are_nan_and_neighbours_use_present_tiles() -> None:
    full = grid()
    holes = {(4, 4), (0, 0)}
    slide = replace(full, tiles=tuple(s for s in full.tiles if (s.row, s.col) not in holes))
    _, result = run(checkerboard, slide)
    a = result.rasters["prob_tumour"].array
    assert np.isnan(a[4, 4]) and np.isnan(a[0, 0])
    np.testing.assert_allclose(a, brute_force(slide, checkerboard), atol=1e-3)
    assert len(result.detections) == 81 - 2


def test_tiles_become_core_objects_with_class_probabilities() -> None:
    _, result = run(checkerboard)
    tiles = result.detections
    assert len(tiles) == 81
    np.testing.assert_allclose(tiles.class_probs.sum(1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(tiles.labels, tiles.class_probs.argmax(1))
    np.testing.assert_allclose(tiles.scores, tiles.class_probs.max(1))
    areas = shapely.area(tiles.geometries("box"))
    assert areas.sum() == pytest.approx(4096 * 4096)  # cores partition the slide
    interior = (STRIDE * DOWNSAMPLE) ** 2
    assert np.sum(np.isclose(areas, interior)) == 7 * 7


def test_threshold_regions_follow_the_checkerboard() -> None:
    _, result = run(checkerboard, thresholds={"tumour": 0.5}, simplify_px=0, smooth=0)
    # The four tumour blocks touch only at corners: one region with 8-connectivity.
    (region,) = result.features
    assert region.class_name == "tumour"
    scale, origin = STRIDE * DOWNSAMPLE, OVERLAP // 2 * DOWNSAMPLE
    cells = [
        shapely.box(
            origin + c * scale,
            origin + r * scale,
            origin + (c + 1) * scale,
            origin + (r + 1) * scale,
        )
        for r in range(9)
        for c in range(9)
        if (r // 3 + c // 3) % 2
    ]
    assert shapely.equals(region.geometry, shapely.union_all(cells))  # exactly the cores
    assert region.measurements["area_um2"] == pytest.approx(36 * scale**2 * SLIDE_MPP**2)
    assert 0.8 < region.measurements["mean_probability"] < 0.9
    four = run(checkerboard, thresholds={"tumour": 0.5}, simplify_px=0, smooth=0, connectivity=4)
    assert len(four[1].features) == 4


def test_default_regions_are_smoothed() -> None:
    _, result = run(checkerboard, thresholds={"tumour": 0.5}, connectivity=4)
    core, origin = STRIDE * DOWNSAMPLE, OVERLAP // 2 * DOWNSAMPLE
    blocks = [
        shapely.box(*(origin + 3 * core * v for v in (c, r, c + 1, r + 1)))
        for r in range(3)
        for c in range(3)
        if (r + c) % 2
    ]
    assert len(result.features) == len(blocks)
    for region in result.features:
        assert len(region.geometry.exterior.coords) > 5  # corners rounded
        # Smoothing works at the scale of one grid pixel (a core here): each region stays
        # inside its exact 3 x 3-core block and covers all of it but the outer core.
        (block,) = [b for b in blocks if b.contains(region.geometry.centroid)]
        assert block.buffer(1).contains(region.geometry)
        assert region.geometry.contains(block.buffer(-core))


def test_smoothing_spreads_but_keeps_probabilities_and_gaps() -> None:
    hot = lambda s: 1.0 if (s.row, s.col) == (4, 4) else 0.0
    full = grid()
    slide = replace(full, tiles=tuple(s for s in full.tiles if (s.row, s.col) != (0, 0)))
    plain = run(hot, slide)[1].rasters
    smooth = run(hot, slide, sigma_tiles=1.0)[1].rasters
    assert smooth["prob_tumour"].array[4, 6] > plain["prob_tumour"].array[4, 6] == 0
    assert smooth["prob_tumour"].array[4, 4] < plain["prob_tumour"].array[4, 4]
    assert np.isnan(smooth["prob_tumour"].array[0, 0])
    total = smooth["prob_tumour"].array.astype(np.float32) + smooth["prob_normal"].array
    np.testing.assert_allclose(total[~np.isnan(total)], 1.0, atol=2e-3)


def test_logits_are_softmaxed() -> None:
    stitcher = HeatmapStitcher(card("logits"), grid())
    specs = list(grid().tiles)
    stitcher.add(specs, {"scores": torch.tensor([[0.0, 0.0]] * len(specs))})
    np.testing.assert_allclose(stitcher.core_probabilities(), 0.5, atol=1e-6)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"thresholds": {"tumor": 0.5}}, "unknown classes"),
        ({"thresholds": {"tumour": 0.0}}, r"\(0, 1\]"),
        ({"sigma_tiles": -1.0}, "sigma_tiles"),
    ],
)
def test_bad_options_are_rejected(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        HeatmapStitcher(card(), grid(), **options)
