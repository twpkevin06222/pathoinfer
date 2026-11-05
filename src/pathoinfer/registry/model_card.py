"""ModelCard: everything pathoinfer needs to run a model besides its weights.

pathoinfer owns this schema. The published JSON Schema is ``schema/model_card.v1.json``,
generated from :class:`ModelCard`; ``docs/model_contract.md`` describes it and how legacy
pathotrain exports map onto it. Lands in v0.1.
"""

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal

import matplotlib
from pydantic import BaseModel, ConfigDict, Field

Task = Literal["segmentation", "detection", "heatmap"]
Byte = Annotated[int, Field(ge=0, le=255)]
RGB = tuple[int, int, int]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ClassInfo(_Strict):
    """One class: its output index (channel or label), name and optional display colour."""

    index: int = Field(ge=0)
    name: str
    color: tuple[Byte, Byte, Byte] | None = None


class Preprocessing(_Strict):
    """Tiles are RGB uint8, read at ``tile_mpp`` (ICC to sRGB when ``icc``), then fed as
    float32 NCHW: ``x / 255``, then ``(x - mean) / std`` per channel."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    icc: bool = True


class OutputSpec(_Strict):
    """One model output. ``shape`` uses names for dynamic axes, e.g. ``["batch", 300, 4]``."""

    name: str
    shape: list[int | str]
    dtype: Literal["float32", "float16", "int64", "int32"]
    kind: Literal["logits", "probabilities", "boxes_xyxy", "scores", "labels"]


class ModelCard(_Strict):
    """A model folder's ``model_card.json``; ``file`` is relative to that folder."""

    schema_version: Literal[1] = 1
    name: str
    version: str
    task: Task
    architecture: str | None = None
    format: Literal["onnx", "torchscript", "torch_export", "tensorrt_plan"]
    file: str
    input_name: str = "image"
    tile_size: int = Field(gt=0)
    tile_mpp: float = Field(gt=0)
    preprocessing: Preprocessing
    classes: list[ClassInfo] = Field(min_length=1)
    outputs: list[OutputSpec] = Field(min_length=1)
    provenance: dict[str, Any] = {}


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
