import json
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import pytest
import shapely
from polars.testing import assert_frame_equal

from pathoinfer.io.objects import ObjectTable, RunInfo
from pathoinfer.io.writers.parquet import (
    META_KEY,
    objects_frame,
    pixel_crs,
    summary_frame,
    write_parquet,
    write_summary,
)


def read_duckdb(path: Path) -> pl.DataFrame:
    with duckdb.connect() as con:
        return con.sql("SELECT * FROM read_parquet($p)", params={"p": str(path)}).pl()


@pytest.mark.parametrize("task", ["seg", "det"])
def test_polars_and_duckdb_read_the_same_table(
    tmp_path: Path, task: str, request: pytest.FixtureRequest
) -> None:
    objects: ObjectTable = request.getfixturevalue(f"{task}_objects")
    run: RunInfo = request.getfixturevalue(f"{task}_run")
    path = write_parquet(tmp_path / "objects.parquet", objects, run)
    by_polars = pl.read_parquet(path)
    assert_frame_equal(by_polars, objects_frame(objects, run))
    assert_frame_equal(read_duckdb(path), by_polars, check_dtypes=False)
    assert by_polars.columns[:6] == [
        "slide_id",
        "object_id",
        "task",
        "class_id",
        "class_name",
        "score",
    ]
    assert by_polars.columns[-3:] == ["geometry", "model_name", "model_version"]
    meta = json.loads(pl.read_parquet_metadata(path)[META_KEY])
    assert meta["model_name"] == run.model_name


def test_wkb_decodes_to_the_same_shapes(
    tmp_path: Path, seg_objects: ObjectTable, seg_run: RunInfo
) -> None:
    path = write_parquet(tmp_path / "objects.parquet", seg_objects, seg_run)
    frame = pl.read_parquet(path)
    decoded = shapely.from_wkb(frame["geometry"].to_numpy())
    assert shapely.equals_exact(decoded, seg_objects.geometry, tolerance=0).all()
    centroids = shapely.centroid(decoded)
    # Stitched polygons keep their 0.1 px precision grid, which snaps their centroids.
    np.testing.assert_allclose(frame["centroid_x"], shapely.get_x(centroids), atol=0.05)
    np.testing.assert_allclose(
        frame.select("bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1").to_numpy(),
        shapely.bounds(decoded),
    )
    assert frame["score"].null_count() == len(frame)


def test_row_groups_hold_about_100k_rows(tmp_path: Path, det_run: RunInfo) -> None:
    n = 250_000
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 4000, (n, 2))
    objects = ObjectTable(
        shapely.points(xy),
        rng.integers(0, 2, n),
        rng.uniform(0.5, 1, n).astype(np.float32),
        np.full(n, 4.0),
        {},
    )
    path = write_parquet(tmp_path / "big.parquet", objects, det_run)
    with duckdb.connect() as con:
        groups = con.sql(
            "SELECT DISTINCT row_group_id, row_group_num_rows FROM parquet_metadata($p)",
            params={"p": str(path)},
        ).fetchall()
    assert sorted(rows for _, rows in groups) == [50_000, 100_000, 100_000]


def test_summary_counts_every_class(
    tmp_path: Path, det_objects: ObjectTable, det_run: RunInfo
) -> None:
    path = write_summary(tmp_path / "summary.parquet", det_objects, det_run)
    summary = pl.read_parquet(path)
    assert summary["class_name"].to_list() == ["tumour", "lymphocyte"]
    expected = np.bincount(det_objects.class_id, minlength=2)
    assert summary["count"].to_list() == expected.tolist()
    assert summary["mean_score"].to_list() == pytest.approx([0.9, 0.9])
    assert (summary["tissue_area_um2"] == det_run.tissue_area_um2).all()
    assert (summary["model_version"] == det_run.model_version).all()


def test_summary_keeps_classes_without_objects(tmp_path: Path, det_run: RunInfo) -> None:
    empty = ObjectTable(
        np.empty(0, object),
        np.empty(0, np.int64),
        np.empty(0, np.float32),
        np.empty(0),
        {},
    )
    summary = pl.read_parquet(write_summary(tmp_path / "s.parquet", empty, det_run))
    assert summary["count"].to_list() == [0, 0]
    assert summary["mean_score"].null_count() == 2


def _spatial() -> bool:
    try:
        with duckdb.connect() as con:
            con.load_extension("spatial")
    except duckdb.Error:
        return False
    return True


@pytest.mark.skipif(not _spatial(), reason="DuckDB spatial extension is not installed")
def test_geoparquet_metadata_names_the_pixel_crs(
    tmp_path: Path, seg_objects: ObjectTable, seg_run: RunInfo
) -> None:
    path = write_parquet(tmp_path / "geo.parquet", seg_objects, seg_run, geoparquet=True)
    geo = json.loads(pl.read_parquet_metadata(path)["geo"])
    column = geo["columns"]["geometry"]
    assert geo["primary_column"] == "geometry" and column["encoding"] == "WKB"
    assert column["crs"] == pixel_crs(seg_run.slide_mpp)
    frame = pl.read_parquet(path)
    decoded = shapely.from_wkb(frame["geometry"].to_numpy())
    assert shapely.equals_exact(decoded, seg_objects.geometry, tolerance=0).all()
    assert_frame_equal(frame, objects_frame(seg_objects, seg_run), check_dtypes=False)


def test_summary_as_json_maps_class_names_to_counts(
    tmp_path: Path, det_objects: ObjectTable, det_run: RunInfo
) -> None:
    path = write_summary(tmp_path / "summary.json", det_objects, det_run)
    summary = json.loads(path.read_text())
    frame = summary_frame(det_objects, det_run)
    assert list(summary["classes"]) == frame["class_name"].to_list()
    for row in frame.iter_rows(named=True):
        assert summary["classes"][row["class_name"]] == {
            k: row[k] for k in ("class_id", "count", "mean_score", "total_area_um2")
        }
    assert summary["slide_id"] == det_run.slide_id
    assert summary["tissue_area_um2"] == det_run.tissue_area_um2
    assert not list(tmp_path.glob("*.partial"))
