import logging
import time
from dataclasses import replace

import numpy as np
import pytest
import shapely
import torch

from pathoinfer.registry.model_card import DetectionCard, parse_card
from pathoinfer.stitch.base import StitchThread
from pathoinfer.stitch.detection import Detections, DetectionStitcher
from pathoinfer.tiling.grid import SlideGrid, TileSpec

from .test_segmentation import CORNER, DOWNSAMPLE, SIZE, SLIDE_MPP, TILE, grid

SLOTS = 300  # detections per tile, as a DETR-style model returns them


def card(
    box_format: str = "xyxy",
    box_units: str = "pixel",
    probs: bool = False,
    overlap: float = 32 / TILE,
) -> DetectionCard:
    output: dict[str, object] = {
        "box_format": box_format,
        "box_units": box_units,
        "score_threshold": 0.5,
    }
    if probs:
        output["class_probs"] = "probs"
    parsed = parse_card(
        {
            "name": "cells",
            "version": "1",
            "task": "detection",
            "framework": "onnx",
            "file": "model.onnx",
            "input": {
                "tile_size": TILE,
                "mpp": 0.5,
                "normalisation": {"mean": [0.5] * 3, "std": [0.5] * 3},
            },
            "classes": [
                {"id": 0, "name": "tumour", "color": [255, 0, 0]},
                {"id": 1, "name": "lymphocyte", "color": [0, 0, 255]},
            ],
            "output": output,
            "recommended_overlap": overlap,
        }
    )
    assert isinstance(parsed, DetectionCard)
    return parsed


def detect(
    specs: list[TileSpec], objects: np.ndarray, labels: np.ndarray, card: DetectionCard
) -> dict[str, torch.Tensor]:
    """An ideal detector: every object overlapping the tile, its box clipped to the tile
    (so cut objects shift their centre), in the card's box format; unused slots score 0."""
    n = len(specs)
    boxes = np.zeros((n, SLOTS, 4), np.float32)
    scores = np.zeros((n, SLOTS), np.float32)
    classes = np.zeros((n, SLOTS), np.int64)
    probs = np.zeros((n, SLOTS, 2), np.float32)
    for t, s in enumerate(specs):
        x0, y0 = s.xm * DOWNSAMPLE, s.ym * DOWNSAMPLE
        x1, y1 = x0 + TILE * DOWNSAMPLE, y0 + TILE * DOWNSAMPLE
        seen = np.flatnonzero(
            (objects[:, 2] > x0)
            & (objects[:, 0] < x1)
            & (objects[:, 3] > y0)
            & (objects[:, 1] < y1)
        )
        assert len(seen) <= SLOTS
        clipped = np.clip(objects[seen], [x0, y0, x0, y0], [x1, y1, x1, y1])
        local = (clipped - [x0, y0, x0, y0]) / DOWNSAMPLE
        if card.output.box_format == "cxcywh":
            local = np.concatenate(
                [(local[:, :2] + local[:, 2:]) / 2, local[:, 2:] - local[:, :2]], 1
            )
        if card.output.box_units == "normalised":
            local = local / TILE
        boxes[t, : len(seen)] = local
        scores[t, : len(seen)] = 0.9
        classes[t, : len(seen)] = labels[seen]
        probs[t, : len(seen), 0] = np.where(labels[seen] == 0, 0.8, 0.2)
        probs[t, : len(seen), 1] = 1 - probs[t, : len(seen), 0]
    outputs = {
        "boxes": torch.from_numpy(boxes),
        "scores": torch.from_numpy(scores),
        "labels": torch.from_numpy(classes),
    }
    if card.output.class_probs:
        outputs["probs"] = torch.from_numpy(probs)
    return outputs


def stitch(
    objects: np.ndarray, card: DetectionCard, slide: SlideGrid | None = None, **options: object
) -> Detections:
    slide = slide or grid()
    labels = np.arange(len(objects)) % 2
    stitcher = DetectionStitcher(card, slide, **options)  # type: ignore[arg-type]
    specs = list(slide.tiles)
    for start in range(0, len(specs), 9):
        chunk = specs[start : start + 9]
        stitcher.add(chunk, detect(chunk, objects, labels, card))
    detections = stitcher.finalize().detections
    assert isinstance(detections, Detections)
    return detections


def squares(centres: np.ndarray, size: float) -> np.ndarray:
    return np.concatenate([centres - size / 2, centres + size / 2], axis=1)


@pytest.mark.parametrize(("box_format", "box_units"), [("xyxy", "pixel"), ("cxcywh", "normalised")])
def test_object_on_a_four_tile_corner_is_reported_once(box_format: str, box_units: str) -> None:
    obj = squares(np.array([[CORNER, CORNER]]), 60)
    found = stitch(obj, card(box_format, box_units))
    assert len(found) == 1
    np.testing.assert_allclose(found.boxes[0], obj[0], atol=1e-3)


def test_objects_straddling_tile_corners_and_edges_are_reported_once() -> None:
    rng = np.random.default_rng(0)
    seams = (np.arange(1, 9) * (TILE - 32) + 16) * DOWNSAMPLE
    corners = np.stack(np.meshgrid(seams, seams), -1).reshape(-1, 2)
    # Objects up to `overlap` model px wide, centred within a few px of every core corner.
    centres = corners + rng.uniform(-20, 20, corners.shape)
    found = stitch(squares(centres, 32 * DOWNSAMPLE), card())
    assert len(found) == len(centres)


def test_dense_grid_gives_the_exact_count() -> None:
    step = 50.0
    axis = np.arange(step / 2, SIZE, step)
    centres = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    objects = squares(centres, 40)
    found = stitch(objects, card())
    assert len(found) == len(objects)
    found_order = np.lexsort(found.centroids().T)
    truth_order = np.lexsort(centres.T)
    np.testing.assert_allclose(found.boxes[found_order], objects[truth_order], atol=1e-3)


def test_nms_removes_duplicates_of_objects_larger_than_the_overlap() -> None:
    rng = np.random.default_rng(1)
    seams = (np.arange(1, 9) * (TILE - 32) + 16) * DOWNSAMPLE
    rows = (np.arange(0, 9) * (TILE - 32) + TILE / 2) * DOWNSAMPLE
    centres = np.stack(
        [rng.choice(seams, 30) + rng.uniform(-10, 10, 30), rng.choice(rows, 30)], axis=1
    )
    objects = squares(centres, 90 * DOWNSAMPLE)  # 90 model px across a 32 px overlap
    assert len(stitch(objects, card())) == 2 * len(objects)
    # Each tile sees the object cut at its edge: IoU is about overlap / width = 0.36.
    assert len(stitch(objects, card(), nms_iou=0.5)) == 2 * len(objects)
    assert len(stitch(objects, card(), nms_iou=0.3)) == len(objects)


def test_nms_never_merges_neighbours_from_the_same_tile() -> None:
    centres = np.array([[CORNER - 30, 1000], [CORNER - 10, 1000]])
    found = stitch(squares(centres, 30), card(), nms_iou=0.01)
    assert len(found) == 2


def test_score_threshold_defaults_to_the_card_and_can_be_overridden() -> None:
    objects = squares(np.array([[1000.0, 1000.0]]), 40)
    assert len(stitch(objects, card())) == 1
    assert len(stitch(objects, card(), score_threshold=0.95)) == 0


def test_low_overlap_is_warned_about(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        DetectionStitcher(card(overlap=0.25), grid())
    assert "below the card's recommended 64 px" in caplog.text


def test_features_carry_geometry_and_measurements() -> None:
    objects = squares(np.array([[1000.0, 1000.0], [2000.0, 2000.0]]), 40)
    found = stitch(objects, card(probs=True))
    names = ["tumour", "lymphocyte"]
    boxes = found.to_features(names, SLIDE_MPP, "box")
    points = found.to_features(names, SLIDE_MPP, "point")
    assert all(f.geometry.geom_type == "Polygon" for f in boxes)
    assert all(f.geometry.geom_type == "Point" for f in points)
    first = next(f for f in boxes if f.class_id == 0)
    assert first.measurements["area_um2"] == pytest.approx(40 * 40 * SLIDE_MPP**2)
    assert first.measurements["prob_tumour"] == pytest.approx(0.8)
    assert first.measurements["score"] == pytest.approx(0.9)
    assert "area_um2" not in points[0].measurements
    assert shapely.equals(points[0].geometry, shapely.centroid(boxes[0].geometry))


def test_stitch_thread_runs_the_detection_stitcher() -> None:
    objects = squares(np.array([[CORNER, CORNER]]), 40)
    slide, labels = grid(), np.zeros(1, np.int64)
    with StitchThread(DetectionStitcher(card(), slide)) as thread:
        for spec in slide.tiles:
            thread.put([spec], detect([spec], objects, labels, card()))
        assert len(thread.finalize().detections) == 1


def test_a_million_detections_finish_in_seconds() -> None:
    slide = grid()
    specs = list(slide.tiles)
    per_tile = 1_000_000 // len(specs) + 1
    rng = np.random.default_rng(2)
    xy = rng.uniform(0, TILE - 8, (len(specs), per_tile, 2)).astype(np.float32)
    outputs = {
        "boxes": torch.from_numpy(np.concatenate([xy, xy + 8], axis=2)),
        "scores": torch.from_numpy(
            rng.uniform(0.5, 1.0, (len(specs), per_tile)).astype(np.float32)
        ),
        "labels": torch.from_numpy(rng.integers(0, 2, (len(specs), per_tile))),
    }
    stitcher = DetectionStitcher(card(), slide, nms_iou=0.5)
    start = time.perf_counter()
    for i in range(0, len(specs), 10):
        stitcher.add(specs[i : i + 10], {k: v[i : i + 10] for k, v in outputs.items()})
    found = stitcher.finalize().detections
    elapsed = time.perf_counter() - start
    assert len(specs) * per_tile >= 1_000_000
    assert 300_000 < len(found) < 1_000_000
    assert elapsed < 10, f"{elapsed:.1f} s"


def test_grid_fixture_matches_the_detector_geometry() -> None:
    slide = grid()
    assert replace(slide, tiles=()).overlap == 32 and slide.downsample == DOWNSAMPLE
