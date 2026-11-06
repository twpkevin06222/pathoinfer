"""ModelCard: everything pathoinfer needs to run a model besides its weights.

pathoinfer owns this schema. A model folder holds the model file and ``model_card.json``.
The card is one of four task variants, discriminated by ``task``, so a detection card
cannot carry a background class or a segmentation output. ``ModelCard`` is the three
tile-model variants every tile code path takes; ``VqaCard`` describes a vision-language
model folder; ``AnyCard`` is either. The published JSON Schema is
``schema/model_card.v1.json``; ``docs/model_contract.md`` documents every field.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, Self

import matplotlib
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

TileTask = Literal["segmentation", "detection", "classification"]
Task = Literal["segmentation", "detection", "classification", "vqa"]
Byte = Annotated[int, Field(ge=0, le=255)]
Positive = Annotated[float, Field(gt=0)]
RGB = tuple[int, int, int]
Framework = Literal["onnx", "torch", "tensorrt", "safetensors"]

# The model file suffixes each framework accepts; ``card create`` infers the framework from them.
FRAMEWORK_SUFFIXES: dict[Framework, tuple[str, ...]] = {
    "onnx": (".onnx",),
    "torch": (".pt", ".pt2", ".pth", ".ts"),
    "tensorrt": (".plan", ".engine"),
    "safetensors": (".safetensors",),
}


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClassInfo(_Strict):
    """One class. ``id`` is its output channel (segmentation, classification) or label
    (detection); ``color`` is its display colour."""

    id: int = Field(ge=0)
    name: str = Field(min_length=1)
    color: tuple[Byte, Byte, Byte]


class Normalisation(_Strict):
    """Per channel, in ``channel_order``: ``(pixel * scale - mean) / std``."""

    mean: tuple[float, float, float]
    std: tuple[Positive, Positive, Positive]
    scale: float = Field(1 / 255, gt=0)


class InputSpec(_Strict):
    """The tile the model takes: ``tile_size`` px square, read at ``mpp``."""

    name: str = "image"
    tile_size: int = Field(gt=0)
    mpp: float = Field(gt=0)
    channel_order: Literal["RGB", "BGR"] = "RGB"
    layout: Literal["NCHW", "NHWC"] = "NCHW"
    dtype: Literal["float32", "float16", "uint8"] = "float32"
    normalisation: Normalisation
    icc: bool = Field(True, description="Convert the slide's ICC profile to sRGB first.")


class SegmentationOutput(_Strict):
    """One ``(batch, n_classes, H, W)`` tensor; channel ``i`` is class id ``i``."""

    tensor: str = "logits"
    activation: Literal["logits", "probs"]
    n_classes: int = Field(gt=0)

    def tensor_names(self) -> tuple[str, ...]:
        return (self.tensor,)


class DetectionOutput(_Strict):
    """Three tensors, one row per detection: boxes ``(batch, n, 4)``, scores ``(batch, n)``
    and labels ``(batch, n)`` holding class ids; optionally ``class_probs``
    ``(batch, n, n_classes)``, each detection's probability per class."""

    boxes: str = "boxes"
    scores: str = "scores"
    labels: str = "labels"
    box_format: Literal["xyxy", "cxcywh"]
    box_units: Literal["pixel", "normalised"]
    score_threshold: float = Field(ge=0, le=1)
    class_probs: str | None = None

    def tensor_names(self) -> tuple[str, ...]:
        names = (self.boxes, self.scores, self.labels)
        return names if self.class_probs is None else (*names, self.class_probs)


class ClassificationOutput(_Strict):
    """One ``(batch, n_classes)`` tensor per tile; column ``i`` is class id ``i``."""

    tensor: str = "logits"
    activation: Literal["logits", "probs"]
    n_classes: int = Field(gt=0)

    def tensor_names(self) -> tuple[str, ...]:
        return (self.tensor,)


class Parity(_Strict):
    """How a converted model matched its full-precision reference through the real
    stitchers. Written only when every metric was within its tolerance."""

    reference: str = Field(description="name@version of the full-precision model compared.")
    tile_set_hash: str = Field(description="Hash of the tiles both models ran on.")
    metrics: dict[str, float] = Field(
        description="Worst value of each metric over classes, e.g. dice_min, f1_min, auc_min."
    )
    tolerances: dict[str, float] = Field(description="The bound each metric had to meet.")
    checked_at: str = Field(description="UTC time of the check, ISO 8601.")


class Provenance(_Strict):
    training_run_id: str | None = None
    calibration_set_hash: str | None = None
    parent_model: str | None = Field(
        None, description="name@version of the model this one was converted from."
    )
    parity: Parity | None = Field(
        None, description="Parity with the reference model, for converted models."
    )
    source_metadata: dict[str, Any] = Field(
        {}, description="Producer metadata kept as is, e.g. a legacy 'pathotrain' ONNX key."
    )


class _Card(_Strict):
    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    framework: Framework
    file: str = Field(min_length=1, description="Model file, relative to the model folder.")
    precision: Literal["fp32", "fp16", "bf16", "int8", "fp8"] = "fp32"
    input: InputSpec
    classes: list[ClassInfo] = Field(min_length=1)
    recommended_overlap: float = Field(ge=0, lt=1)
    provenance: Provenance = Provenance()

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [c.id for c in self.classes]
        if ids != list(range(len(ids))):
            raise ValueError(f"class ids must be 0 .. {len(ids) - 1} in order, got {ids}")
        names = [c.name for c in self.classes]
        if len(set(names)) != len(names):
            raise ValueError(f"class names must be unique, got {names}")
        if not self.file.endswith(FRAMEWORK_SUFFIXES[self.framework]):
            raise ValueError(
                f"{self.framework} model file must end with "
                f"{' or '.join(FRAMEWORK_SUFFIXES[self.framework])}, got {self.file!r}"
            )
        n_classes = getattr(self.output, "n_classes", len(ids))  # type: ignore[attr-defined]
        if n_classes != len(ids):
            raise ValueError(f"output has {n_classes} classes, card lists {len(ids)}")
        background = getattr(self, "background_id", None)
        if background is not None and background not in ids:
            raise ValueError(f"background_id {background} is not a class id {ids}")
        return self


class SegmentationCard(_Card):
    task: Literal["segmentation"]
    background_id: int | None = Field(
        description="Class id of background, never written as an annotation; null if none."
    )
    output: SegmentationOutput


class DetectionCard(_Card):
    """Every detection label is a foreground class, so there is no background id."""

    task: Literal["detection"]
    output: DetectionOutput


class ClassificationCard(_Card):
    task: Literal["classification"]
    background_id: int | None = Field(
        description="Class id of tiles that are not a finding (e.g. normal); null if none."
    )
    output: ClassificationOutput


class VqaInput(_Strict):
    """How the slide crops a VLM sees are made: ``tile_size`` px squares read at ``mpp``
    (ICC-converted when ``icc``); ``max_pixels`` caps what the processor resizes to."""

    tile_size: int = Field(gt=0)
    mpp: float = Field(gt=0)
    max_pixels: int | None = Field(None, gt=0)
    icc: bool = True


class VqaOutput(_Strict):
    """``grounding``: the model answers with boxes, in ``box_format`` and ``box_units``
    (``per_mille``: 0-1000 of the image, as Qwen-VL; ``normalised``: 0-1; ``pixel``)."""

    grounding: bool = False
    box_format: Literal["xyxy", "cxcywh"] = "xyxy"
    box_units: Literal["pixel", "normalised", "per_mille"] = "per_mille"


class VqaCard(_Strict):
    """A vision-language model answering questions about slide crops, as a Hugging Face
    model folder (``config.json``, safetensors shards, processor files) that vLLM serves.
    ``precision`` is the language model's weights (``int4`` for W4A16); the vision
    encoder and projector stay bf16."""

    schema_version: Literal[1] = 1
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    task: Literal["vqa"]
    framework: Literal["hf"] = "hf"
    file: Literal["config.json"] = "config.json"
    architecture: str = Field(
        min_length=1, description="transformers class, e.g. Qwen3VLForConditionalGeneration."
    )
    precision: Literal["bf16", "fp16", "fp8", "int8", "int4"] = "bf16"
    input: VqaInput
    output: VqaOutput = VqaOutput()
    provenance: Provenance = Provenance()


ModelCard = Annotated[
    SegmentationCard | DetectionCard | ClassificationCard, Field(discriminator="task")
]
AnyCard = Annotated[
    SegmentationCard | DetectionCard | ClassificationCard | VqaCard, Field(discriminator="task")
]
CARD: TypeAdapter[ModelCard] = TypeAdapter(ModelCard)
ANY_CARD: TypeAdapter[AnyCard] = TypeAdapter(AnyCard)


def parse_card(data: object) -> ModelCard:
    """Validate a tile-model card from parsed JSON. Raises pydantic.ValidationError."""
    return CARD.validate_python(data)


def parse_any_card(data: object) -> AnyCard:
    """Validate any card, tile model or VLM. Raises pydantic.ValidationError."""
    return ANY_CARD.validate_python(data)


def card_json_schema() -> dict[str, Any]:
    """The JSON Schema published as ``schema/model_card.v1.json``."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ModelCard",
        **ANY_CARD.json_schema(),
    }


def class_colors(
    names: Sequence[str], overrides: Mapping[str, RGB] | None = None, cmap: str = "tab20"
) -> dict[str, RGB]:
    """RGB colour per class name: ``cmap``'s colours in class order, wrapping after
    its last, with ``overrides`` replacing any of them.

    Raises:
        ValueError: ``overrides`` names a class the model does not have.
    """
    overrides = overrides or {}
    unknown = set(overrides) - set(names)
    if unknown:
        raise ValueError(f"colours for unknown classes {sorted(unknown)}; classes: {list(names)}")
    # shortcut: cycling suits qualitative maps (tab10, tab20); a continuous map gives
    # near-identical neighbours, so sample it evenly if users ask for one.
    colormap = matplotlib.colormaps[cmap]
    colors = {}
    for i, name in enumerate(names):
        r, g, b, _ = colormap(i % colormap.N)
        colors[name] = (round(r * 255), round(g * 255), round(b * 255))
    return colors | dict(overrides)
