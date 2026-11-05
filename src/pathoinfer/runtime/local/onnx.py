"""ONNX Runtime: TensorRT EP, then CUDA EP, then CPU, whichever loads first. Lands in v0.1."""

from pathlib import Path

from pathoinfer._extras import extra, require
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Runtime


def open_session(card: ModelCard, folder: Path) -> Runtime:
    require("onnxruntime", extra("onnx"))
    raise NotImplementedError("lands in v0.1")
