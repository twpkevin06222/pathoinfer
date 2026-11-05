"""Parquet tables for analysis: one row per object, and one summary row per class.

The objects table is a Polars frame built straight from NumPy columns. Geometry is WKB in
level-0 pixels. Every file also stores the run metadata as JSON under the Parquet
key-value metadata key ``pathoinfer``.

``geoparquet=True`` writes through DuckDB's spatial extension instead, which adds the
GeoParquet ``geo`` metadata. Without a ``crs`` that metadata would mean longitude and
latitude (OGC:CRS84), so the geometry is tagged with a PROJJSON engineering CRS for
level-0 pixels, y down. DuckDB writes the metadata; every write reads it back and fails
if the ``geo`` entry is missing or names another CRS.
"""

import json
import os
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import shapely
from loguru import logger

from pathoinfer._logging import log_call
from pathoinfer.io.objects import ObjectTable, RunInfo

ROW_GROUP = 100_000
META_KEY = "pathoinfer"


def objects_frame(objects: ObjectTable, run: RunInfo) -> pl.DataFrame:
    """The objects table as written to Parquet."""
    bounds, centroids = objects.bounds(), objects.centroids()
    names = np.array([""] * (max((c.id for c in run.classes), default=-1) + 1), dtype=object)
    for c in run.classes:
        names[c.id] = c.name
    columns: dict[str, Any] = {
        "slide_id": pl.repeat(run.slide_id, len(objects), eager=True),
        "object_id": np.arange(len(objects), dtype=np.int64),
        "task": pl.repeat(run.task, len(objects), eager=True),
        "class_id": objects.class_id,
        "class_name": pl.Series(names[objects.class_id], dtype=pl.String),
        "score": pl.Series(objects.score).fill_nan(None),
        "centroid_x": centroids[:, 0],
        "centroid_y": centroids[:, 1],
        "bbox_x0": bounds[:, 0],
        "bbox_y0": bounds[:, 1],
        "bbox_x1": bounds[:, 2],
        "bbox_y1": bounds[:, 3],
        "area_um2": objects.area_um2,
        **objects.measurements,
        "geometry": pl.Series(shapely.to_wkb(objects.geometry), dtype=pl.Binary),
        "model_name": pl.repeat(run.model_name, len(objects), eager=True),
        "model_version": pl.repeat(run.model_version, len(objects), eager=True),
    }
    return pl.DataFrame(columns)


@log_call
def write_parquet(
    path: str | Path, objects: ObjectTable, run: RunInfo, geoparquet: bool = False
) -> Path:
    """Write the objects table (zstd, statistics, ~100k-row row groups)."""
    path = Path(path)
    frame = objects_frame(objects, run)
    partial = path.with_name(path.name + ".partial")
    if geoparquet:
        _write_geoparquet(partial, frame, run)
    else:
        frame.write_parquet(
            partial,
            compression="zstd",
            statistics=True,
            row_group_size=ROW_GROUP,
            metadata={META_KEY: json.dumps(run.to_dict())},
        )
    os.replace(partial, path)
    logger.info("wrote {} ({} objects)", path, len(objects))
    return path


def summary_frame(objects: ObjectTable, run: RunInfo) -> pl.DataFrame:
    """One row per class, zero-count classes included: ``count``, ``mean_score`` (null
    without scores), ``total_area_um2``, the slide's ``tissue_area_um2`` and the run
    metadata."""
    per_object = pl.DataFrame(
        {
            "class_id": objects.class_id,
            "score": pl.Series(objects.score).fill_nan(None),
            "area_um2": objects.area_um2,
        }
    )
    stats = per_object.group_by("class_id").agg(
        pl.len().alias("count"),
        pl.col("score").mean().alias("mean_score"),
        pl.col("area_um2").sum().alias("total_area_um2"),
    )
    meta = {k: v for k, v in run.to_dict().items() if k != "classes"}
    first = ["slide_id", "class_id", "class_name", "count", "mean_score", "total_area_um2"]
    return (
        pl.DataFrame(
            {"class_id": [c.id for c in run.classes], "class_name": [c.name for c in run.classes]},
            schema={"class_id": pl.Int64, "class_name": pl.String},
        )
        .join(stats, on="class_id", how="left")
        .with_columns(
            pl.col("count").fill_null(0).cast(pl.UInt32),
            pl.col("total_area_um2").fill_null(0.0),
            **{k: pl.lit(v) for k, v in meta.items()},
        )
        .select(*first, *(k for k in meta if k not in first))
    )


@log_call
def write_summary(path: str | Path, objects: ObjectTable, run: RunInfo) -> Path:
    path = Path(path)
    summary_frame(objects, run).write_parquet(
        path, compression="zstd", metadata={META_KEY: json.dumps(run.to_dict())}
    )
    logger.info("wrote {}", path)
    return path


def pixel_crs(slide_mpp: float) -> dict[str, Any]:
    """PROJJSON for level-0 pixels: x right, y down, one unit = ``slide_mpp`` microns."""
    unit = {"type": "LinearUnit", "name": "level-0 pixel", "conversion_factor": slide_mpp * 1e-6}
    return {
        "type": "EngineeringCRS",
        "name": "Slide level-0 pixels",
        "datum": {"type": "EngineeringDatum", "name": "Slide level-0 top-left corner"},
        "coordinate_system": {
            "subtype": "Cartesian",
            "axis": [
                {"name": "x", "abbreviation": "x", "direction": "east", "unit": unit},
                {"name": "y", "abbreviation": "y", "direction": "south", "unit": unit},
            ],
        },
    }


def _write_geoparquet(path: Path, frame: pl.DataFrame, run: RunInfo) -> None:
    con = duckdb.connect()
    try:
        con.load_extension("spatial")
    except duckdb.Error:
        con.install_extension("spatial")
        con.load_extension("spatial")
    crs = pixel_crs(run.slide_mpp)
    con.register("objects", frame.to_arrow())
    con.sql(
        "COPY (SELECT * REPLACE (ST_SetCRS(ST_GeomFromWKB(geometry), $crs) AS geometry) "
        "FROM objects) TO '" + str(path).replace("'", "''") + "' (FORMAT parquet, "
        f"COMPRESSION zstd, ROW_GROUP_SIZE {ROW_GROUP}, KV_METADATA {{{META_KEY}: $meta}})",
        params={"crs": json.dumps(crs), "meta": json.dumps(run.to_dict())},
    )
    con.close()
    geo = json.loads(pl.read_parquet_metadata(path).get("geo", "{}"))
    column = geo.get("columns", {}).get("geometry", {})
    if (
        geo.get("primary_column") != "geometry"
        or column.get("encoding") != "WKB"
        or (column.get("crs") != crs)
    ):
        path.unlink()
        raise RuntimeError(
            f"DuckDB {duckdb.__version__} did not write the expected GeoParquet metadata "
            f"(got {geo}); write with geoparquet=False"
        )
