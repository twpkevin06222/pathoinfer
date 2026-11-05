from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pathoinfer.analysis import class_density, score_histogram
from pathoinfer.io.objects import ObjectTable, RunInfo
from pathoinfer.io.writers.parquet import write_parquet, write_summary


@pytest.fixture
def two_slides(tmp_path: Path, det_objects: ObjectTable, det_run: RunInfo) -> Path:
    tumour_only = det_objects.take(np.flatnonzero(det_objects.class_id == 0))
    for name, objects in [("a", det_objects), ("b", tumour_only)]:
        run = replace(det_run, slide_id=name, slide_name=f"slide_name-{name}.svs")
        write_parquet(tmp_path / f"{name}.objects.parquet", objects, run)
        write_summary(tmp_path / f"{name}.summary.parquet", objects, run)
    return tmp_path


def test_density_per_class_per_slide(
    two_slides: Path, det_objects: ObjectTable, det_run: RunInfo
) -> None:
    density = class_density(two_slides / "*.objects.parquet", two_slides / "*.summary.parquet")
    counts = np.bincount(det_objects.class_id, minlength=2)
    assert density.select("slide_id", "class_name", "count").rows() == [
        ("a", "tumour", counts[0]),
        ("a", "lymphocyte", counts[1]),
        ("b", "tumour", counts[0]),
        ("b", "lymphocyte", 0),
    ]
    mm2 = det_run.tissue_area_um2 / 1e6
    assert density["per_mm2"].to_list() == pytest.approx([c / mm2 for c in density["count"]])


def test_min_score_filters_before_counting(two_slides: Path) -> None:
    density = class_density(
        two_slides / "*.objects.parquet", two_slides / "*.summary.parquet", min_score=0.95
    )
    assert density["count"].sum() == 0


def test_score_histogram_counts_every_scored_object(
    two_slides: Path, det_objects: ObjectTable
) -> None:
    hist = score_histogram(two_slides / "*.objects.parquet", bins=10)
    a = hist.filter(slide_id="a")
    assert a["count"].sum() == len(det_objects)
    assert set(a["bin"]) == {9} and a["lower"][0] == pytest.approx(0.9)


def test_min_score_keeps_objects_without_scores(
    tmp_path: Path, seg_objects: ObjectTable, seg_run: RunInfo
) -> None:
    write_parquet(tmp_path / "s.objects.parquet", seg_objects, seg_run)
    write_summary(tmp_path / "s.summary.parquet", seg_objects, seg_run)
    density = class_density(
        tmp_path / "*.objects.parquet", tmp_path / "*.summary.parquet", min_score=0.5
    )
    assert density["count"].to_list() == [len(seg_objects)]
