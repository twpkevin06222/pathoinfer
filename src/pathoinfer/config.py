"""Run settings: one YAML file per run, validated here (examples in ``examples/``).

Relative paths (model, slides, output folder) are relative to the working directory.
"""

import glob
import hashlib
import json
from pathlib import Path
from typing import Literal, NamedTuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from patchology import ArtifactConfig, Backend, TissueConfig
from pathoinfer.registry.model_card import Byte, SegmentationCard, TileTask


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
    ``fp16``; otherwise a model runs at its card's precision. ``triton_url`` (gRPC, e.g.
    ``triton:8001``) runs the model on a Triton server instead, under ``triton_model``
    (default: the card's name); the model folder is still read for its card."""

    device: str = Field("auto", pattern=r"^(auto|cpu|cuda(:\d+)?)$")
    batch_size: int = Field(32, gt=0)
    tensorrt: bool = True
    autocast: Literal["bf16", "fp16"] | None = None
    triton_url: str | None = None
    triton_model: str | None = None


class LoaderConfig(_Strict):
    """Tile reading. ``num_workers`` 0 reads in the main process."""

    num_workers: int = Field(4, ge=0)
    prefetch_factor: int = Field(2, gt=0)


class SegmentationOptions(_Strict):
    """Segmentation stitching (``stitch/README.md``). ``output_mpp`` None polygonises at
    2.0 mpp, or at the model's mpp when that is coarser. ``smooth`` rounds of Chaikin
    corner cutting round off pixel steps (each doubles the vertices); 0 keeps outlines
    on pixel edges. Holes under ``fill_holes_um2`` are filled (unless another class lies
    inside) and regions under ``min_area_um2`` dropped; ``.inf`` fills every hole.
    ``connectivity`` 8 keeps parts touching only at a pixel corner in one region (a
    MultiPolygon meeting at that point); 4 makes them separate regions.

    ``background_classes`` (names) count as background. ``class_map`` renames and merges:
    each key is an output class name, its value the model class it renames or a list of
    model classes it merges, e.g. ``{tumor: invasive tumor, stroma: [inflamed stroma,
    tumor-associated stroma]}``. Classes not named keep their names. Probabilities move
    before each pixel's class is chosen: a pixel 30% stroma, 30% inflamed stroma and 40%
    tumour becomes stroma (60%) when the two stromas merge. Every output then shows the
    result: label map, polygons, summary, overlay."""

    output_mpp: float | None = Field(None, gt=0)
    simplify_px: float = Field(1.5, ge=0)
    smooth: int = Field(2, ge=0, le=5)
    min_area_um2: float = Field(0.0, ge=0)
    fill_holes_um2: float = Field(0.0, ge=0)
    connectivity: Literal[4, 8] = 8
    background_classes: list[str] = []
    class_map: dict[str, str | list[str]] = {}

    def class_remap(self, card: SegmentationCard) -> "ClassRemap":
        """What ``background_classes`` and ``class_map`` do to ``card``'s classes, by id.
        A merged class keeps the id of its lowest-numbered source. Raises ValueError for
        unknown or empty sources, a class used twice or named as the background, background
        classes without a background, and output names that clash."""
        ids = {c.name: c.id for c in card.classes}
        groups = {
            new: [old] if isinstance(old, str) else old for new, old in self.class_map.items()
        }
        empty = sorted(new for new, old in groups.items() if not old)
        if empty:
            raise ValueError(f"class_map entries without a source class: {empty}")
        sources = [*self.background_classes, *(n for old in groups.values() for n in old)]
        unknown = sorted({n for n in sources if n not in ids})
        if unknown:
            raise ValueError(
                f"segmentation names unknown classes {unknown}; the model has {sorted(ids)}"
            )
        twice = sorted({n for n in sources if sources.count(n) > 1})
        if twice:
            raise ValueError(
                f"classes used more than once in background_classes and class_map: {twice}"
            )
        background = card.background_id
        if background is not None and any(ids[n] == background for n in sources):
            raise ValueError("background_classes and class_map cannot use the background class")
        if self.background_classes and background is None:
            raise ValueError("background_classes needs a model with a background class")
        unmapped = [c.name for c in card.classes if c.name not in sources]
        names = [*groups, *unmapped]
        clash = sorted({n for n in names if names.count(n) > 1})
        if clash:
            raise ValueError(f"class_map output names clash with other classes: {clash}")
        targets = {ids[n]: background for n in self.background_classes if background is not None}
        rename = {}
        for new, old in groups.items():
            keep, *rest = sorted(ids[n] for n in old)
            targets |= {i: keep for i in rest}
            rename[keep] = new
        return ClassRemap(targets, rename)


class ClassRemap(NamedTuple):
    """``targets``: class id -> the id its probability moves to. ``names``: id -> its new
    name, for the classes ``class_map`` names."""

    targets: dict[int, int]
    names: dict[int, str]


class DetectionOptions(_Strict):
    """Detection stitching. ``score_threshold`` None uses the card's. ``nms_iou`` None
    turns cross-tile NMS off. ``geometry`` is what writers output per detection."""

    score_threshold: float | None = Field(None, ge=0, le=1)
    nms_iou: float | None = Field(None, gt=0, le=1)
    geometry: Literal["point", "box"] = "box"


class OutputConfig(_Strict):
    """What to write per slide into ``dir`` (``docs/outputs.md``). ``parquet`` writes the
    objects table and the per-class summary, as Parquet or ``summary_format: json``.
    ``tiff`` writes the segmentation label map; ``overlay`` a PNG of the slide at
    ``overlay_mpp`` with the segmentation polygons filled in their class colours."""

    dir: Path = Path("outputs")
    geojson: bool = True
    gzip: bool = True
    per_class: bool = False
    parquet: bool = True
    summary_format: Literal["parquet", "json"] = "parquet"
    geoparquet: bool = False
    tiff: bool = False
    overlay: bool = True
    overlay_mpp: float = Field(6.0, gt=0)


class RunConfig(_Strict):
    """One run. ``slides`` is a path, a glob, a list file (``.txt`` one path per line,
    ``.csv`` or ``.parquet`` with a ``path`` column), or a list of any of these.
    ``resume`` skips slides whose outputs exist from a run with the same settings.
    ``class_colors`` replaces the colour of any class, by class name."""

    task: TileTask
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
        return expand_slides(self.slides)


def expand_slides(slides: str | list[str]) -> list[Path]:
    """Paths from a path, a glob, a list file or a list of these, duplicates removed."""
    entries = [slides] if isinstance(slides, str) else slides
    return list(dict.fromkeys(p for entry in entries for p in _expand(entry)))


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


class CalibrationConfig(_Strict):
    """Calibration tiles, read through patchology at the model's tile size and mpp.
    List slides across scanners, stains and tissue types; tiles are spread evenly over
    them."""

    slides: str | list[str]
    tiles: int = Field(512, gt=0)
    seed: int = 0
    backend: Backend = "openslide"


class QuantConfig(_Strict):
    """Static post-training quantisation (``pathoinfer convert ptq``).

    ``engine`` ``modelopt`` (NVIDIA ModelOpt) targets TensorRT, keeps excluded layers in
    FP16 and does INT8 and FP8; ``ort`` (ONNX Runtime) does INT8 that runs in ONNX
    Runtime too, keeping excluded layers in FP32. Kept in high precision: nodes whose op
    type is in ``keep_ops``, whose name matches a glob in ``keep_nodes`` (lower case), and
    with ``keep_ends`` the first and last Conv/MatMul/Gemm. The defaults suit CNNs and
    ViT backbones (LayerNorm, Softmax, attention)."""

    precision: Literal["int8", "fp8"] = "int8"
    engine: Literal["modelopt", "ort"] = "modelopt"
    method: Literal["entropy", "minmax", "percentile"] = "entropy"
    percentile: float = Field(99.999, gt=50, lt=100)
    calibration: CalibrationConfig
    keep_ops: list[str] = ["LayerNormalization", "Softmax"]
    keep_nodes: list[str] = ["*attn*", "*attention*"]
    keep_ends: bool = True

    @model_validator(mode="after")
    def _fp8_needs_modelopt(self) -> "QuantConfig":
        if self.precision == "fp8" and self.engine != "modelopt":
            raise ValueError("fp8 needs engine: modelopt")
        return self


class ParityTolerances(_Strict):
    """Bounds a converted model must meet against its reference, per task: segmentation
    Dice per class (polygon level), detection F1 per class and relative count difference,
    classification one-vs-rest AUC per class (the reference's argmax as labels)."""

    dice_min: float = Field(0.97, ge=0, le=1)
    f1_min: float = Field(0.97, ge=0, le=1)
    count_diff_max: float = Field(0.03, ge=0)
    auc_min: float = Field(0.99, ge=0, le=1)


class ParityConfig(_Strict):
    """``pathoinfer convert parity``: both models run on the same mini-slides (windows of
    ``window_tiles`` x ``window_tiles`` grid tiles around tissue, ``windows`` per slide)
    through the real stitchers, with this config's stitch settings."""

    slides: str | list[str]
    window_tiles: int = Field(6, gt=0)
    windows: int = Field(2, gt=0)
    seed: int = 0
    backend: Backend = "openslide"
    reference_device: str = Field("auto", pattern=r"^(auto|cpu|cuda(:\d+)?)$")
    candidate: RuntimeConfig = RuntimeConfig()
    iou: float = Field(0.5, gt=0, le=1)
    tolerances: ParityTolerances = ParityTolerances()
    segmentation: SegmentationOptions = SegmentationOptions()
    detection: DetectionOptions = DetectionOptions()


class VqaDataConfig(_Strict):
    """Image + question pairs: a ``.jsonl``, ``.csv`` or ``.parquet`` table with columns
    ``image`` (a path, relative to the table's folder) and ``question``, plus ``answer``
    and optionally ``box`` (``[x0, y0, x1, y1]`` in image pixels) for evaluation. Pathology
    crops at the model's tile size and mpp, not generic images. ``samples`` None uses
    every row; otherwise a seeded random subset."""

    path: Path
    samples: int | None = Field(256, gt=0)
    seed: int = 0


class VlmQuantConfig(_Strict):
    """``pathoinfer convert vlm``: llm-compressor recipe for the language model only.

    ``fp8``: W8A8 FP8 (dynamic per-token activations; no calibration data needed).
    ``w4a16-awq`` / ``w4a16-gptq``: 4-bit weights, 16-bit activations, calibrated on
    ``calibration`` (a few hundred pathology image + question pairs). The vision encoder
    and projector stay bf16, as does ``lm_head``; ``extra_ignore`` adds module patterns."""

    scheme: Literal["fp8", "w4a16-awq", "w4a16-gptq"] = "fp8"
    calibration: VqaDataConfig | None = None
    max_seq_len: int = Field(2048, gt=0)
    extra_ignore: list[str] = []

    @model_validator(mode="after")
    def _calibration_when_needed(self) -> "VlmQuantConfig":
        if self.scheme != "fp8" and self.calibration is None:
            raise ValueError(f"{self.scheme} needs calibration data")
        return self


class VlmParityTolerances(_Strict):
    """Bounds on how much worse the candidate may be than the bf16 reference."""

    accuracy_drop_max: float = Field(0.02, ge=0, le=1)
    iou_drop_max: float = Field(0.05, ge=0, le=1)


class VlmParityConfig(_Strict):
    """``pathoinfer convert vlm-parity``: both models answer ``eval`` (rows with
    ``answer``; ``box`` for grounding models) through OpenAI-compatible vLLM servers."""

    eval: VqaDataConfig
    tolerances: VlmParityTolerances = VlmParityTolerances()
    max_tokens: int = Field(64, gt=0)
    concurrency: int = Field(8, gt=0)
    timeout_s: float = Field(120.0, gt=0)


def load_yaml[T: BaseModel](path: str | Path, model: type[T]) -> T:
    with open(path) as f:
        return model.model_validate(yaml.safe_load(f))
