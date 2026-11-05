"""Build a TensorRT plan per GPU from an ONNX model folder. Lands in v0.2."""

from pathlib import Path

from pathoinfer._extras import extra, require


def build_plan(folder: Path, out: Path) -> Path:
    require("tensorrt", extra("trt"))
    raise NotImplementedError("lands in v0.2")
