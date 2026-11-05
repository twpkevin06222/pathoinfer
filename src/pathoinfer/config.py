"""Inference settings, one YAML file per task (examples in ``configs/``)."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from patchology import ArtifactConfig, Backend, TissueConfig

Task = Literal["segmentation", "detection", "heatmap"]
Byte = Annotated[int, Field(ge=0, le=255)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PatchologyInput(_Strict):
    """How patchology opens, masks and tiles a slide.

    Tile size and mpp come from the model. There is deliberately no
    ``slide_mpp`` override: a slide without a resolution in its metadata is
    refused by patchology, never inferred on.
    """

    backend: Backend = "openslide"
    icc: bool = True
    tile_overlap: float = Field(0.0, ge=0, lt=1)
    tissue_overlap: float = Field(0.1, ge=0, le=1)
    tissue: TissueConfig = TissueConfig()
    artifacts: ArtifactConfig = ArtifactConfig()


class InferenceConfig(_Strict):
    """One task's settings. ``model`` is an ``.onnx`` path or ``models:/NAME/<version>``.
    ``class_colors`` replaces the default colour of any class, by class name."""

    task: Task
    model: str
    patchology: PatchologyInput = PatchologyInput()
    class_colors: dict[str, tuple[Byte, Byte, Byte]] = {}


def load_config(path: str | Path) -> InferenceConfig:
    with open(path) as f:
        return InferenceConfig.model_validate(yaml.safe_load(f))
