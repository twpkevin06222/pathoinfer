"""GeoJSON FeatureCollections that QuPath imports.

Coordinates are level-0 pixels (x right, y down), rounded to 0.1 px: the convention of
pathology viewers, not RFC 7946's WGS84 longitude and latitude. Each feature carries
QuPath's properties ``objectType``, ``classification {name, color}``, ``isLocked`` and
``measurements``, plus ``object_id`` (the row in the Parquet table). The collection's
``properties`` hold the run metadata (a foreign member, which QuPath ignores).

Features are streamed to disk, so a million detections never sit in memory as JSON.
``.geojson.gz`` is gzip-compressed. The file appears only once complete.
"""

import gzip
import json
import math
import os
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import shapely
from loguru import logger
from numpy.typing import NDArray

from pathoinfer._logging import log_call
from pathoinfer.io.objects import ObjectTable, RunInfo

SUFFIXES = (".geojson", ".geojson.gz")
_CHUNK = 10_000


@log_call
def write_geojson(
    path: str | Path, objects: ObjectTable, run: RunInfo, per_class: bool = False
) -> list[Path]:
    """Write ``objects`` to ``path`` (``.geojson`` or ``.geojson.gz``).

    ``per_class`` writes one file per class with objects instead, named
    ``<stem>.<class>.geojson[.gz]``. Returns the files written.
    """
    path = Path(path)
    suffix = next((s for s in SUFFIXES if path.name.endswith(s)), None)
    if suffix is None:
        raise ValueError(f"{path}: expected a .geojson or .geojson.gz file")
    if not per_class:
        _write(path, objects, run, np.arange(len(objects)))
        return [path]
    stem = path.name.removesuffix(suffix)
    names = {c.id: c.name for c in run.classes}
    written = []
    for class_id in np.unique(objects.class_id):
        name = re.sub(r"[^\w.-]+", "_", names[int(class_id)])
        out = path.with_name(f"{stem}.{name}{suffix}")
        _write(out, objects, run, np.flatnonzero(objects.class_id == class_id))
        written.append(out)
    return written


def _write(path: Path, objects: ObjectTable, run: RunInfo, rows: NDArray[np.intp]) -> None:
    partial = path.with_name(path.name + ".partial")
    with (
        gzip.open(partial, "wt", compresslevel=6, encoding="utf-8")
        if path.name.endswith(".gz")
        else open(partial, "w", encoding="utf-8")
    ) as f:
        f.write('{"type":"FeatureCollection","properties":')
        f.write(json.dumps(run.to_dict()))
        f.write(',"features":[')
        for start in range(0, len(rows), _CHUNK):
            chunk = rows[start : start + _CHUNK]
            for i, feature in enumerate(_features(objects.take(chunk), chunk, run)):
                f.write(feature if start + i == 0 else ",\n" + feature)
        f.write("]}\n")
    os.replace(partial, path)
    logger.info("wrote {} ({} objects)", path, len(rows))


def _features(objects: ObjectTable, ids: NDArray[np.intp], run: RunInfo) -> Iterator[str]:
    """One JSON string per object; ``ids`` are their ``object_id``s."""
    rounded = shapely.transform(objects.geometry, lambda c: np.round(c, 1))
    geometries = shapely.to_geojson(rounded)
    classification = {
        c.id: json.dumps({"name": c.name, "color": list(c.color)}) for c in run.classes
    }
    columns: dict[str, NDArray[np.floating]] = {}
    if run.task == "detection":
        columns["score"] = objects.score
    columns |= {"area_um2": objects.area_um2, **objects.measurements}
    names = list(columns)
    rows = zip(*(columns[k].tolist() for k in names), strict=True)
    head = f'{{"objectType":"{run.object_type}","isLocked":false,"classification":'
    for geometry, class_id, object_id, values in zip(
        geometries, objects.class_id.tolist(), ids.tolist(), rows, strict=True
    ):
        # JSON has no NaN; QuPath reads a missing measurement as NaN.
        measurements = json.dumps(
            {k: v for k, v in zip(names, values, strict=True) if not math.isnan(v)}
        )
        yield (
            f'{{"type":"Feature","geometry":{geometry},"properties":{head}'
            f'{classification[class_id]},"measurements":{measurements},'
            f'"object_id":{object_id}}}}}'
        )
