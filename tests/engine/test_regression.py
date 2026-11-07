"""Pinned results on the synthetic slide. A change here means outputs changed: confirm it
is intended, then update the numbers in the same commit and say why.

``tiny_dark`` labels every pixel darker than glass, so its polygon is the ellipse drawn
by ``write_pyramid`` (centre (1500, 1400), semi-axes 900 x 700 level-0 px at 0.25 mpp);
it is also checked against the analytic shape. Default settings (``simplify_px`` 1.5,
``smooth`` 2): regions are the exact union of their pixels, and smoothing keeps area and perimeter within
0.2% of the ellipse (the unsmoothed pixel outline's perimeter is 14% long). The rectangle that ``write_pyramid`` also
draws is not in patchology's tissue mask today, so it has no tiles and no polygon.
"""

import math
from pathlib import Path

import polars as pl
import pytest
import shapely

from pathoinfer.config import RunConfig
from pathoinfer.engine.pipeline import run_batch
from tests.engine.test_pipeline import FIXTURES, config


def summary(cfg: RunConfig) -> list[tuple[str, int, float]]:
    frame = pl.read_parquet(cfg.outputs.dir / "slide_name-01.summary.parquet")
    return frame.select("class_name", "count", "total_area_um2").rows()


def test_segmentation_fixture(tmp_path: Path, slides: Path) -> None:
    cfg = config(tmp_path, slides, "segmentation")
    (result,) = run_batch(cfg)
    assert (result.tiles, result.objects) == (75, 1)
    assert summary(cfg) == [("tumor", 1, pytest.approx(187608.0, rel=1e-9))]


def test_detection_fixture(tmp_path: Path, slides: Path) -> None:
    cfg = config(tmp_path, slides, "detection")
    (result,) = run_batch(cfg)
    assert (result.tiles, result.objects) == (75, 75)
    assert summary(cfg) == [("lymphocyte", 29, 29696.0), ("tumour", 46, 47104.0)]


def test_tissue_polygon_matches_the_drawn_ellipse(tmp_path: Path, slides: Path) -> None:
    cfg = RunConfig.model_validate(
        {
            "task": "segmentation",
            "model": str(FIXTURES / "tiny_dark"),
            "slides": str(slides / "slide_name-01.tiff"),
            "runtime": {"device": "cpu", "batch_size": 64},
            "loader": {"num_workers": 0},
            "segmentation": {"output_mpp": 1.0},
            "outputs": {"dir": str(tmp_path / "out")},
        }
    )
    (result,) = run_batch(cfg)
    assert (result.tiles, result.objects) == (249, 1)
    (row,) = pl.read_parquet(cfg.outputs.dir / "slide_name-01.objects.parquet").rows(named=True)
    # Re-pinned when borders became shared arcs (post.vectorise.shape_borders): a closed
    # outline is now simplified from its smallest vertex, not from where its ring started.
    pinned = {
        "area_um2": 123840.20187499996,
        "perimeter_um": 1263.8565569817004,
        "centroid_x": 1500.7,
        "centroid_y": 1400.7,
        "bbox_x0": 600.2,
        "bbox_y0": 700.2,
        "bbox_x1": 2400.0,
        "bbox_y1": 2099.9,
    }
    assert {k: row[k] for k in pinned} == pytest.approx(pinned, rel=1e-9)
    # Against the drawn shape: OpenCV draws on pixel centres, so the centre is (1500.5, 1400.5).
    assert row["area_um2"] == pytest.approx(math.pi * 900 * 700 * 0.25**2, rel=0.005)
    a, b = 900, 700  # Ramanujan's ellipse perimeter, level-0 px
    perimeter = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b))) * 0.25
    assert row["perimeter_um"] == pytest.approx(perimeter, rel=0.005)
    assert row["centroid_x"] == pytest.approx(1500.5, abs=0.5)
    assert row["centroid_y"] == pytest.approx(1400.5, abs=0.5)
    # Smoothed, then simplified at 0.1 output px: curved, but no redundant vertices.
    assert len(shapely.from_wkb(row["geometry"]).exterior.coords) == 67  # 63 before shared arcs
