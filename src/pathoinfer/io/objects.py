"""What every writer consumes: the run's metadata and the slide's objects as columns.

``RunInfo`` is recorded in every output. ``ObjectTable`` holds one row per object as
NumPy columns, built from a stitcher's ``StitchResult``; detections never become Python
objects on the way. Coordinates are level-0 pixels, x right and y down.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from typing import Any, Literal

import numpy as np
import shapely
from numpy.typing import NDArray

from pathoinfer.registry.model_card import RGB, ClassInfo, ModelCard, Task
from pathoinfer.stitch.base import StitchResult
from pathoinfer.tiling.grid import SlideGrid

ObjectType = Literal["annotation", "detection"]
# QuPath object type per task: regions are annotations, cells are detections.
OBJECT_TYPES: dict[Task, ObjectType] = {
    "segmentation": "annotation",
    "detection": "detection",
    "classification": "annotation",
}


@dataclass(frozen=True)
class RunInfo:
    """One slide's run. ``classes`` are the classes an object can have (the card's minus
    its ``background_id``), with any colour overrides applied."""

    slide_id: str
    slide_name: str
    task: Task
    model_name: str
    model_version: str
    model_source: str
    model_card_schema: int
    slide_mpp: float
    model_mpp: float
    output_mpp: float | None
    tile_size: int
    overlap: int
    tissue_area_um2: float
    classes: tuple[ClassInfo, ...]
    pathoinfer_version: str
    patchology_version: str
    created_at: str

    @classmethod
    def create(
        cls,
        card: ModelCard,
        grid: SlideGrid,
        model_source: str,
        output_mpp: float | None = None,
        colors: Mapping[str, RGB] | None = None,
    ) -> "RunInfo":
        """Raises ValueError if ``colors`` names a class the card does not have."""
        colors = colors or {}
        unknown = set(colors) - {c.name for c in card.classes}
        if unknown:
            raise ValueError(f"colours for unknown classes {sorted(unknown)}")
        background = getattr(card, "background_id", None)
        mask = grid.tissue_mask
        tissue_px = float(mask.array.sum()) * mask.downsample[0] * mask.downsample[1]
        return cls(
            slide_id=grid.slide_id,
            slide_name=grid.slide_path.name,
            task=card.task,
            model_name=card.name,
            model_version=card.version,
            model_source=model_source,
            model_card_schema=card.schema_version,
            slide_mpp=grid.slide_mpp,
            model_mpp=grid.mpp,
            output_mpp=output_mpp,
            tile_size=grid.tile_size,
            overlap=grid.overlap,
            tissue_area_um2=tissue_px * grid.slide_mpp**2,
            classes=tuple(
                c.model_copy(update={"color": colors.get(c.name, c.color)})
                for c in card.classes
                if c.id != background
            ),
            pathoinfer_version=version("pathoinfer"),
            patchology_version=version("patchology"),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    @property
    def object_type(self) -> ObjectType:
        return OBJECT_TYPES[self.task]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready metadata; ``classes`` as ``{id, name, color}`` objects."""
        out = asdict(self)
        out["classes"] = [c.model_dump(mode="json") for c in self.classes]
        return out


@dataclass(frozen=True)
class ObjectTable:
    """One row per object. ``score`` is NaN for tasks without one (segmentation);
    ``measurements`` are extra float columns in output order."""

    geometry: NDArray[np.object_]
    class_id: NDArray[np.int64]
    score: NDArray[np.float32]
    area_um2: NDArray[np.float64]
    measurements: dict[str, NDArray[np.float64]]

    def __len__(self) -> int:
        return len(self.class_id)

    def take(self, rows: NDArray[np.intp]) -> "ObjectTable":
        return ObjectTable(
            self.geometry[rows],
            self.class_id[rows],
            self.score[rows],
            self.area_um2[rows],
            {k: v[rows] for k, v in self.measurements.items()},
        )

    @classmethod
    def from_result(
        cls,
        result: StitchResult,
        run: RunInfo,
        geometry: Literal["point", "box"] = "box",
    ) -> "ObjectTable":
        """``geometry`` picks box polygons or centre points for detections. A detection's
        ``area_um2`` is its box area either way."""
        if result.detections is not None:
            d = result.detections
            size = d.boxes[:, 2:] - d.boxes[:, :2]
            measurements = {}
            if d.class_probs is not None:
                classes = sorted(run.classes, key=lambda c: c.id)
                measurements = {
                    f"prob_{c.name}": d.class_probs[:, i].astype(np.float64)
                    for i, c in enumerate(classes)
                }
            return cls(
                d.geometries(geometry),
                d.labels.astype(np.int64),
                d.scores.astype(np.float32),
                size.prod(axis=1) * run.slide_mpp**2,
                measurements,
            )
        features = result.features
        keys = [k for k in (features[0].measurements if features else {}) if k != "area_um2"]
        return cls(
            np.array([f.geometry for f in features] or [], dtype=object),
            np.array([f.class_id for f in features], np.int64),
            np.full(len(features), np.nan, np.float32),
            np.array([f.measurements.get("area_um2", np.nan) for f in features], np.float64),
            {k: np.array([f.measurements[k] for f in features], np.float64) for k in keys},
        )

    def bounds(self) -> NDArray[np.float64]:
        """``(n, 4)`` ``x0, y0, x1, y1`` per object."""
        return np.asarray(shapely.bounds(self.geometry)).reshape(-1, 4)

    def centroids(self) -> NDArray[np.float64]:
        return np.asarray(shapely.get_coordinates(shapely.centroid(self.geometry))).reshape(-1, 2)
