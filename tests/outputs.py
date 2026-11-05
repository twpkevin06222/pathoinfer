"""Stitched results and their tables, shared by the writer and analysis tests."""

import numpy as np
import pytest

from pathoinfer.io.objects import ObjectTable, RunInfo
from pathoinfer.stitch.base import StitchResult

from .stitch import test_detection as det
from .stitch import test_segmentation as seg


@pytest.fixture(scope="session")
def seg_result() -> StitchResult:
    """A disk on a four-tile corner and an annulus: one polygon with a hole."""
    disk, ring = seg.disk(300), seg.annulus(100, 200)
    return seg.stitch(lambda x, y: disk(x - 1200, y) | ring(x + 1200, y))


@pytest.fixture(scope="session")
def seg_run() -> RunInfo:
    return RunInfo.create(seg.card(), seg.grid(), "models/tiny_seg", output_mpp=seg.OUTPUT_MPP)


@pytest.fixture(scope="session")
def det_run() -> RunInfo:
    return RunInfo.create(det.card(probs=True), seg.grid(), "models:/cells@prod")


@pytest.fixture(scope="session")
def det_result() -> StitchResult:
    axis = np.arange(100.0, seg.SIZE, 400.0)
    centres = np.stack(np.meshgrid(axis, axis), -1).reshape(-1, 2)
    return StitchResult([], detections=det.stitch(det.squares(centres, 40), det.card(probs=True)))


@pytest.fixture(scope="session")
def seg_objects(seg_result: StitchResult, seg_run: RunInfo) -> ObjectTable:
    return ObjectTable.from_result(seg_result, seg_run)


@pytest.fixture(scope="session")
def det_objects(det_result: StitchResult, det_run: RunInfo) -> ObjectTable:
    return ObjectTable.from_result(det_result, det_run)
