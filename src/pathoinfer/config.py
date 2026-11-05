"""Inference settings, one YAML file per task (examples in ``configs/``)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from patchology import ArtifactConfig, Backend, TissueConfig
from pathoinfer.registry.model_card import Byte, Task


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PatchologyInput(_Strict):
    """How patchology opens, masks and tiles a slide.

    Tile size and mpp come from the model. ``tile_overlap`` None uses the model card's
    ``recommended_overlap``. There is deliberately no
    ``slide_mpp`` override: a slide without a resolution in its metadata is
    refused by patchology, never inferred on.
    """

    backend: Backend = "openslide"
    icc: bool = True
    tile_overlap: float | None = Field(None, ge=0, lt=1)
    tissue_overlap: float = Field(0.1, ge=0, le=1)
    tissue: TissueConfig = TissueConfig()
    artifacts: ArtifactConfig = ArtifactConfig()


class RuntimeConfig(_Strict):
    """How the model executes. ``device`` is ``auto`` (CUDA when available), ``cpu``,
    ``cuda`` or ``cuda:N``. ``tensorrt`` lets ONNX models use the TensorRT EP when its
    libraries load. ``autocast`` applies to PyTorch models only."""

    device: str = Field("auto", pattern=r"^(auto|cpu|cuda(:\d+)?)$")
    batch_size: int = Field(32, gt=0)
    tensorrt: bool = True
    autocast: Literal["bf16", "fp16"] | None = None


class DetectionOptions(_Strict):
    """Detection stitching. ``score_threshold`` None uses the card's. ``nms_iou`` None
    turns cross-tile NMS off. ``geometry`` is what writers output per detection."""

    score_threshold: float | None = Field(None, ge=0, le=1)
    nms_iou: float | None = Field(None, gt=0, le=1)
    geometry: Literal["point", "box"] = "box"


class InferenceConfig(_Strict):
    """One task's settings. ``model`` is an ``.onnx`` path or ``models:/NAME/<version>``.
    ``class_colors`` replaces the default colour of any class, by class name."""

    task: Task
    model: str
    patchology: PatchologyInput = PatchologyInput()
    runtime: RuntimeConfig = RuntimeConfig()
    detection: DetectionOptions = DetectionOptions()
    class_colors: dict[str, tuple[Byte, Byte, Byte]] = {}


def load_config(path: str | Path) -> InferenceConfig:
    with open(path) as f:
        return InferenceConfig.model_validate(yaml.safe_load(f))
