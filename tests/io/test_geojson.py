import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import shapely

from pathoinfer.io.objects import ObjectTable, RunInfo
from pathoinfer.io.writers.geojson import write_geojson


def load(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as f:
        data: dict[str, Any] = json.load(f)
    return data


def shapes(data: dict[str, Any]) -> list[Any]:
    return [shapely.from_geojson(json.dumps(f["geometry"])) for f in data["features"]]


@pytest.mark.parametrize("name", ["seg.geojson", "seg.geojson.gz"])
def test_segmentation_reads_back_as_the_same_polygons(
    tmp_path: Path, name: str, seg_objects: ObjectTable, seg_run: RunInfo
) -> None:
    (path,) = write_geojson(tmp_path / name, seg_objects, seg_run)
    data = load(path)
    assert data["type"] == "FeatureCollection"
    assert data["properties"]["slide_id"] == "synthetic"
    assert data["properties"]["output_mpp"] == seg_run.output_mpp
    read = shapes(data)
    assert len(read) == len(seg_objects)
    for got, want in zip(read, seg_objects.geometry, strict=True):
        assert got.is_valid and shapely.equals_exact(got, want, tolerance=0.05 + 1e-9)
    assert [len(g.interiors) for g in read] == [len(g.interiors) for g in seg_objects.geometry]
    props = data["features"][0]["properties"]
    assert props["objectType"] == "annotation" and props["isLocked"] is False
    assert props["classification"] == {"name": "tumor", "color": list(seg_run.classes[0].color)}
    measurements = props["measurements"]
    assert set(measurements) == {"score", "area_um2", "perimeter_um", "mean_probability"}
    assert measurements["score"] == pytest.approx(measurements["mean_probability"], rel=1e-6)
    assert not list(tmp_path.glob("*.partial"))


def test_coordinates_have_one_decimal(
    tmp_path: Path, det_objects: ObjectTable, det_run: RunInfo
) -> None:
    (path,) = write_geojson(tmp_path / "det.geojson", det_objects, det_run)
    coords = shapely.get_coordinates(shapes(load(path)))
    np.testing.assert_allclose(coords, np.round(coords, 1), atol=1e-9)


def test_detections_carry_score_and_probabilities(
    tmp_path: Path, det_objects: ObjectTable, det_run: RunInfo
) -> None:
    (path,) = write_geojson(tmp_path / "det.geojson", det_objects, det_run)
    features = load(path)["features"]
    assert len(features) == len(det_objects)
    props = features[0]["properties"]
    assert props["objectType"] == "detection"
    assert set(props["measurements"]) == {"score", "area_um2", "prob_tumour", "prob_lymphocyte"}
    assert [f["properties"]["object_id"] for f in features] == list(range(len(features)))


def test_per_class_files_split_objects_and_keep_object_ids(
    tmp_path: Path, det_objects: ObjectTable, det_run: RunInfo
) -> None:
    paths = write_geojson(tmp_path / "det.geojson.gz", det_objects, det_run, per_class=True)
    assert sorted(p.name for p in paths) == [
        "det.lymphocyte.geojson.gz",
        "det.tumour.geojson.gz",
    ]
    ids = []
    for path in paths:
        features = load(path)["features"]
        names = {f["properties"]["classification"]["name"] for f in features}
        assert names == {path.name.split(".")[1]}
        ids += [f["properties"]["object_id"] for f in features]
    assert sorted(ids) == list(range(len(det_objects)))


def test_other_extensions_are_refused(
    tmp_path: Path, det_objects: ObjectTable, det_run: RunInfo
) -> None:
    with pytest.raises(ValueError, match=".geojson"):
        write_geojson(tmp_path / "det.json", det_objects, det_run)
