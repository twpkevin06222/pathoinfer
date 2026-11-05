"""DuckDB queries over many slides' Parquet outputs, returned as Polars frames.

Arguments are paths or globs, e.g. ``"out/*.objects.parquet"`` and
``"out/*.summary.parquet"``. Files are read by column name, so slides from different
models can share one query.
"""

from pathlib import Path

import duckdb
import polars as pl

from pathoinfer._logging import log_call


@log_call
def class_density(
    objects: str | Path, summaries: str | Path, min_score: float | None = None
) -> pl.DataFrame:
    """Objects per class per slide, and per mm² of tissue. Classes with no objects get 0.
    ``min_score`` drops scored objects (detections) below it; unscored objects
    (segmentation) always count."""
    return _query(
        """
        WITH counts AS (
            SELECT slide_id, class_id, count(*) AS count
            FROM read_parquet($objects, union_by_name = true)
            WHERE $min_score IS NULL OR score IS NULL OR score >= $min_score
            GROUP BY ALL
        )
        SELECT s.slide_id, s.slide_name, s.class_id, s.class_name,
               coalesce(c.count, 0) AS count,
               s.tissue_area_um2 / 1e6 AS tissue_area_mm2,
               coalesce(c.count, 0) / (s.tissue_area_um2 / 1e6) AS per_mm2
        FROM read_parquet($summaries, union_by_name = true) s
        LEFT JOIN counts c USING (slide_id, class_id)
        ORDER BY s.slide_id, s.class_id
        """,
        objects=str(objects),
        summaries=str(summaries),
        min_score=min_score,
    )


@log_call
def score_histogram(objects: str | Path, bins: int = 10) -> pl.DataFrame:
    """Objects per score bin, per slide and class: ``bins`` equal bins over [0, 1]
    (a score of 1 falls in the last). Objects without a score are skipped."""
    return _query(
        """
        SELECT slide_id, class_name,
               least(floor(score * $bins), $bins - 1)::INTEGER AS bin,
               bin / $bins AS lower, (bin + 1) / $bins AS upper,
               count(*) AS count
        FROM read_parquet($objects, union_by_name = true)
        WHERE score IS NOT NULL
        GROUP BY ALL
        ORDER BY slide_id, class_name, bin
        """,
        objects=str(objects),
        bins=bins,
    )


def _query(sql: str, **params: object) -> pl.DataFrame:
    with duckdb.connect() as con:
        frame: pl.DataFrame = con.sql(sql, params=params).pl()
        return frame
