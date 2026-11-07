import json
import math
import random
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import shapely
import torch
from shapely.geometry import Polygon

from patchology import SlideMask
from pathoinfer.registry.model_card import SegmentationCard, parse_card
from pathoinfer.stitch.base import StitchResult, StitchThread
from pathoinfer.stitch.segmentation import SegmentationStitcher, gaussian_window
from pathoinfer.tiling.grid import SlideGrid, TileSpec

SLIDE_MPP, MODEL_MPP, OUTPUT_MPP = 0.25, 0.5, 1.0
TILE, OVERLAP, SIZE = 256, 32, 4096
DOWNSAMPLE = MODEL_MPP / SLIDE_MPP
CARD = json.loads((Path(__file__).parents[1] / "fixtures/tiny_seg/model_card.json").read_text())
# Where tiles (3, 3), (3, 4), (4, 3) and (4, 4) meet, in level-0 pixels.
CORNER = (4 * (TILE - OVERLAP) + OVERLAP // 2) * DOWNSAMPLE

Shape = Callable[[np.ndarray, np.ndarray], np.ndarray]


def card() -> SegmentationCard:
    data = CARD | {"input": CARD["input"] | {"tile_size": TILE, "mpp": MODEL_MPP}}
    parsed = parse_card(data)
    assert isinstance(parsed, SegmentationCard)
    return parsed


def grid() -> SlideGrid:
    stride, model_side = TILE - OVERLAP, SIZE / DOWNSAMPLE
    n = max(1, math.ceil((model_side - TILE) / stride) + 1)
    base = SlideGrid(
        slide_path=Path("synthetic"),
        slide_id="synthetic",
        backend="none",
        slide_dimensions=(SIZE, SIZE),
        slide_mpp=SLIDE_MPP,
        mpp=MODEL_MPP,
        tile_size=TILE,
        overlap=OVERLAP,
        icc=False,
        downsample=DOWNSAMPLE,
        level=0,
        read_factor=1.0,
        n_rows=n,
        n_cols=n,
        tiles=(),
        tissue_mask=SlideMask.from_array(np.ones((2, 2), bool), (SIZE, SIZE)),
    )
    return replace(base, tiles=tuple(base.spec(r, c) for r in range(n) for c in range(n)))


def logits(specs: list[TileSpec], shape: Shape) -> torch.Tensor:
    """Class 1 where ``shape`` (level-0 x, y -> bool) holds at each model pixel centre."""
    centres = (np.arange(TILE) + 0.5) * DOWNSAMPLE
    out = np.empty((len(specs), 2, TILE, TILE), np.float32)
    for i, s in enumerate(specs):
        inside = shape(s.xm * DOWNSAMPLE + centres[None, :], s.ym * DOWNSAMPLE + centres[:, None])
        out[i, 1] = np.where(inside, 8.0, -8.0)
        out[i, 0] = -out[i, 1]
    return torch.from_numpy(out)


def stitch(shape: Shape, order: str = "rows", batch: int = 7, **options: object) -> StitchResult:
    stitcher = SegmentationStitcher(card(), grid(), output_mpp=OUTPUT_MPP, **options)  # type: ignore[arg-type]
    specs = list(grid().tiles)
    if order == "shuffled":
        random.Random(0).shuffle(specs)
    for start in range(0, len(specs), batch):
        chunk = specs[start : start + batch]
        stitcher.add(chunk, {"logits": logits(chunk, shape)})
    return stitcher.finalize()


def disk(r: float) -> Shape:
    return lambda x, y: (x - CORNER) ** 2 + (y - CORNER) ** 2 <= r**2


def annulus(r_in: float, r_out: float) -> Shape:
    return lambda x, y: (
        ((x - CORNER) ** 2 + (y - CORNER) ** 2 <= r_out**2)
        & ((x - CORNER) ** 2 + (y - CORNER) ** 2 > r_in**2)
    )


def blobs(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    ellipse = ((x - 1000) / 600) ** 2 + ((y - 3400) / 400) ** 2 <= 1
    bar = (abs(x - 3000) < 250) & (abs(y - 1800) < 1300)
    return ellipse | bar | annulus(400, 700)(x, y)


def test_disk_on_a_four_tile_corner_is_one_polygon_with_the_right_area() -> None:
    radius = 1200
    result = stitch(disk(radius), band_height=50)
    assert len(result.features) == 1
    feature = result.features[0]
    assert isinstance(feature.geometry, Polygon) and feature.geometry.is_valid
    analytic = math.pi * (radius * SLIDE_MPP) ** 2
    assert feature.measurements["area_um2"] == pytest.approx(analytic, rel=0.02)
    assert feature.measurements["perimeter_um"] == pytest.approx(
        2 * math.pi * radius * SLIDE_MPP, rel=0.02
    )
    assert feature.measurements["mean_probability"] > 0.99
    assert feature.class_name == "tumor"
    assert shapely.centroid(feature.geometry).distance(shapely.Point(CORNER, CORNER)) < 4


def test_annulus_keeps_one_interior_ring() -> None:
    result = stitch(annulus(500, 1200), band_height=50)
    assert len(result.features) == 1
    polygon = result.features[0].geometry
    assert isinstance(polygon, Polygon) and len(polygon.interiors) == 1
    analytic = math.pi * (1200**2 - 500**2) * SLIDE_MPP**2
    assert result.features[0].measurements["area_um2"] == pytest.approx(analytic, rel=0.02)


def same_features(a: StitchResult, b: StitchResult, tolerance: float) -> None:
    assert len(a.features) == len(b.features)
    union_a = shapely.unary_union([f.geometry for f in a.features])
    union_b = shapely.unary_union([f.geometry for f in b.features])
    assert union_a.symmetric_difference(union_b).area <= tolerance * union_a.area
    probs_a = sorted(f.measurements["mean_probability"] for f in a.features)
    probs_b = sorted(f.measurements["mean_probability"] for f in b.features)
    assert probs_a == pytest.approx(probs_b, abs=1e-6)


def test_band_streaming_matches_one_full_canvas() -> None:
    full = stitch(blobs, band_height=100_000, simplify_px=0)
    banded = stitch(blobs, band_height=37, simplify_px=0)
    assert len(full.features) == 3
    same_features(banded, full, tolerance=1e-4)


def test_simplified_band_streaming_stays_within_the_simplify_tolerance() -> None:
    same_features(stitch(blobs, band_height=37), stitch(blobs, band_height=100_000), 1e-2)


def test_arrival_order_does_not_change_the_result() -> None:
    same_features(
        stitch(blobs, order="shuffled", band_height=37), stitch(blobs, band_height=37), 1e-9
    )


def test_band_memory_stays_bounded() -> None:
    stitcher = SegmentationStitcher(card(), grid(), output_mpp=OUTPUT_MPP, band_height=50)
    specs = list(grid().tiles)
    stitcher.add(specs, {"logits": logits(specs[:1], disk(1)).repeat(len(specs), 1, 1, 1)})
    full_canvas = 3 * (SIZE // 4) ** 2 * 4
    assert 0 < stitcher.peak_bytes < full_canvas / 2
    stitcher.finalize()


def test_label_raster_has_a_level0_transform() -> None:
    result = stitch(disk(1200), keep_labels=True)
    labels = result.rasters["labels"]
    assert labels.scale == OUTPUT_MPP / SLIDE_MPP
    row = col = round(CORNER / labels.scale)
    assert labels.array[row, col] == 1 and labels.array[0, 0] == 0


def test_label_raster_shows_the_same_clean_up_as_the_polygons() -> None:
    ring, speck = annulus(20, 300), disk(12)  # hole: 79 um2; speck: 28 um2
    shape = lambda x, y: ring(x, y) | speck(x + 900, y)

    def label_at(result: StitchResult, x: float) -> int:
        labels = result.rasters["labels"]
        return int(labels.array[int(CORNER / labels.scale), int(x / labels.scale)])

    raw = stitch(shape, keep_labels=True)
    assert (label_at(raw, CORNER), label_at(raw, CORNER - 900)) == (0, 1)
    clean = stitch(shape, keep_labels=True, fill_holes_um2=500, min_area_um2=50)
    assert (label_at(clean, CORNER), label_at(clean, CORNER - 900)) == (1, 0)
    (feature,) = clean.features  # the ring, without its hole; the speck is gone
    assert len(feature.geometry.interiors) == 0
    count, _ = cv2.connectedComponents((clean.rasters["labels"].array == 1).astype(np.uint8))
    assert count - 1 == len(clean.features)


def test_output_mpp_finer_than_the_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="finer"):
        SegmentationStitcher(card(), grid(), output_mpp=0.25)


def test_gaussian_window_is_centre_heavy_and_positive() -> None:
    window = gaussian_window(64)
    assert window.max() == pytest.approx(1.0) and window.min() > 0
    assert window[32, 32] > window[32, 8] > window[0, 0]


def test_stitch_thread_gives_the_same_result_as_a_direct_call() -> None:
    specs = list(grid().tiles)
    with StitchThread(
        SegmentationStitcher(card(), grid(), output_mpp=OUTPUT_MPP), maxsize=2
    ) as thread:
        for start in range(0, len(specs), 7):
            chunk = specs[start : start + 7]
            thread.put(chunk, {"logits": logits(chunk, disk(1200))})
        result = thread.finalize()
    same_features(result, stitch(disk(1200)), 1e-9)


def test_stitch_thread_reraises_stitcher_errors() -> None:
    class Broken:
        def add(self, specs: list[TileSpec], outputs: dict[str, torch.Tensor]) -> None:
            raise ValueError("boom")

        def finalize(self) -> StitchResult:
            return StitchResult([])

    with StitchThread(Broken(), maxsize=1) as thread:
        thread.put([], {})
        with pytest.raises(RuntimeError, match="stitcher failed"):
            for _ in range(10):
                thread.put([], {})
            thread.finalize()


def holes(result: StitchResult) -> list[int]:
    return sorted(len(f.geometry.interiors) for f in result.features)


def test_holes_under_the_threshold_are_filled_and_larger_ones_kept() -> None:
    # Hole areas: r 20 -> 79 um2, r 150 -> 4418 um2 (level-0 px at 0.25 mpp).
    small, large = annulus(20, 300), annulus(150, 300)
    shape = lambda x, y: small(x - 900, y) | large(x + 900, y)
    assert holes(stitch(shape)) == [1, 1]
    filled = stitch(shape, fill_holes_um2=500)
    assert holes(filled) == [0, 1]
    unfilled = {round(f.measurements["area_um2"]) for f in stitch(shape).features}
    grown = {round(f.measurements["area_um2"]) for f in filled.features}
    assert len(unfilled & grown) == 1  # the large-hole region is unchanged


def test_islands_inside_a_filled_hole_are_dropped() -> None:
    ring, island = annulus(40, 300), disk(10)
    bullseye = lambda x, y: ring(x, y) | island(x, y)
    assert len(stitch(bullseye).features) == 2
    (filled,) = stitch(bullseye, fill_holes_um2=1000).features
    assert len(filled.geometry.interiors) == 0
    assert filled.geometry.area == pytest.approx(math.pi * 300**2, rel=0.02)


def test_a_hole_holding_another_class_is_kept() -> None:
    data = CARD | {
        "input": CARD["input"] | {"tile_size": TILE, "mpp": MODEL_MPP},
        "classes": [*CARD["classes"], {"id": 2, "name": "necrosis", "color": [0, 0, 200]}],
        "output": CARD["output"] | {"n_classes": 3},
    }
    three = parse_card(data)
    assert isinstance(three, SegmentationCard)
    ring, core = annulus(40, 300), disk(25)
    stitcher = SegmentationStitcher(three, grid(), output_mpp=OUTPUT_MPP, fill_holes_um2=1000)
    centres = (np.arange(TILE) + 0.5) * DOWNSAMPLE
    specs = list(grid().tiles)
    for start in range(0, len(specs), 16):
        chunk = specs[start : start + 16]
        out = np.full((len(chunk), 3, TILE, TILE), -8.0, np.float32)
        for i, s in enumerate(chunk):
            x = s.xm * DOWNSAMPLE + centres[None, :]
            y = s.ym * DOWNSAMPLE + centres[:, None]
            label = np.where(core(x, y), 2, np.where(ring(x, y), 1, 0))
            np.put_along_axis(out[i], label[None], 8.0, axis=0)
        stitcher.add(chunk, {"logits": torch.from_numpy(out)})
    features = stitcher.finalize().features
    by_class = {f.class_name: f for f in features}
    assert set(by_class) == {"tumor", "necrosis"}
    assert len(by_class["tumor"].geometry.interiors) == 1
    assert not by_class["tumor"].geometry.overlaps(by_class["necrosis"].geometry)


def test_small_objects_are_removed() -> None:
    big, speck = disk(300), disk(12)  # speck: ~28 um2
    shape = lambda x, y: big(x, y) | speck(x + 900, y)
    assert len(stitch(shape).features) == 2
    (kept,) = stitch(shape, min_area_um2=100).features
    assert kept.measurements["area_um2"] > 1000
