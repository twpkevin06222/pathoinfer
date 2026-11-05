"""Run settings: one YAML file per run, validated here (examples in ``examples/``).

Relative paths (model, slides, output folder) are relative to the working directory.
"""

import glob
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from patchology import ArtifactConfig, Backend, TissueConfig
from pathoinfer.registry.model_card import Byte, Task


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TilingConfig(_Strict):
    """How patchology opens, masks and tiles a slide.

    Tile size and mpp come from the model. ``overlap`` is a fraction of the tile; None uses
    the card's ``recommended_overlap``. A tile is kept when its tissue fraction is above 0
    and at least ``min_tissue_fraction``. There is deliberately no slide mpp override: a
    slide without a resolution in its metadata is refused, never inferred on.
    """

    backend: Backend = "openslide"
    icc: bool = True
    overlap: float | None = Field(None, ge=0, lt=1)
    min_tissue_fraction: float = Field(0.0, ge=0, le=1)
    tissue: TissueConfig = TissueConfig()
    artifacts: ArtifactConfig = ArtifactConfig()


class RuntimeConfig(_Strict):
    """How the model executes, in this process. ``device`` is ``auto`` (CUDA when
    available), ``cpu``, ``cuda`` or ``cuda:N``. ``tensorrt`` lets ONNX models use the
    TensorRT EP when its libraries load. ``autocast`` runs PyTorch models in ``bf16`` or
    ``fp16``; otherwise a model runs at its card's precision."""

    device: str = Field("auto", pattern=r"^(auto|cpu|cuda(:\d+)?)$")
    batch_size: int = Field(32, gt=0)
    tensorrt: bool = True
    autocast: Literal["bf16", "fp16"] | None = None


class LoaderConfig(_Strict):
    """Tile reading. ``num_workers`` 0 reads in the main process."""

    num_workers: int = Field(4, ge=0)
    prefetch_factor: int = Field(2, gt=0)


class SegmentationOptions(_Strict):
    """Segmentation stitching (``stitch/README.md``). ``output_mpp`` None polygonises at
    2.0 mpp, or at the model's mpp when that is coarser. ``smooth`` rounds of Chaikin
    corner cutting round off pixel steps (each doubles the vertices); 0 keeps outlines
    on pixel edges. Holes under ``fill_holes_um2`` are filled (unless another class lies
    inside) and regions under ``min_area_um2`` dropped; ``.inf`` fills every hole."""

    output_mpp: float | None = Field(None, gt=0)
    simplify_px: float = Field(1.5, ge=0)
    smooth: int = Field(2, ge=0, le=5)
    min_area_um2: float = Field(0.0, ge=0)
    fill_holes_um2: float = Field(0.0, ge=0)


class DetectionOptions(_Strict):
    """Detection stitching. ``score_threshold`` None uses the card's. ``nms_iou`` None
    turns cross-tile NMS off. ``geometry`` is what writers output per detection."""

    score_threshold: float | None = Field(None, ge=0, le=1)
    nms_iou: float | None = Field(None, gt=0, le=1)
    geometry: Literal["point", "box"] = "box"


class OutputConfig(_Strict):
    """What to write per slide into ``dir`` (``docs/outputs.md``). ``tiff`` writes the
    segmentation label map."""

    dir: Path = Path("outputs")
    geojson: bool = True
    gzip: bool = True
    per_class: bool = False
    parquet: bool = True
    geoparquet: bool = False
    tiff: bool = False


class RunConfig(_Strict):
    """One run. ``slides`` is a path, a glob, a list file (``.txt`` one path per line,
    ``.csv`` or ``.parquet`` with a ``path`` column), or a list of any of these.
    ``resume`` skips slides whose outputs exist from a run with the same settings.
    ``class_colors`` replaces the colour of any class, by class name."""

    task: Task
    model: str
    slides: str | list[str]
    tiling: TilingConfig = TilingConfig()
    runtime: RuntimeConfig = RuntimeConfig()
    loader: LoaderConfig = LoaderConfig()
    segmentation: SegmentationOptions = SegmentationOptions()
    detection: DetectionOptions = DetectionOptions()
    outputs: OutputConfig = OutputConfig()
    class_colors: dict[str, tuple[Byte, Byte, Byte]] = {}
    resume: bool = True

    def settings_hash(self, card_json: str) -> str:
        """Hash of everything that changes a slide's outputs: these settings minus the
        slide list, reading and device, plus the model's card."""
        settings = self.model_dump(
            mode="json",
            exclude={
                "slides": True,
                "resume": True,
                "loader": True,
                "runtime": {"device", "batch_size"},
            },
        )
        blob = json.dumps({"settings": settings, "card": json.loads(card_json)}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def slide_paths(self) -> list[Path]:
        """Every slide the ``slides`` entry names, in order, duplicates removed."""
        entries = [self.slides] if isinstance(self.slides, str) else self.slides
        paths = [p for entry in entries for p in _expand(entry)]
        return list(dict.fromkeys(paths))


def _expand(entry: str) -> list[Path]:
    path = Path(entry)
    if path.suffix in (".txt", ".csv", ".parquet") and path.is_file():
        import polars as pl

        if path.suffix == ".parquet":
            column = pl.read_parquet(path, columns=["path"])["path"]
        elif path.suffix == ".csv":
            column = pl.read_csv(path, columns=["path"])["path"]
        else:
            column = pl.read_csv(path, has_header=False, new_columns=["path"])["path"]
        return [path.parent / p for p in column.drop_nulls().str.strip_chars() if p]
    if glob.has_magic(entry):
        return sorted(Path(p) for p in glob.glob(entry, recursive=True))
    return [path]


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        return RunConfig.model_validate(yaml.safe_load(f))
