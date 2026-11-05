"""TensorRT plan runtime: deserialises a plan built by convert.tensorrt for this GPU. Lands in
v0.2.
"""

from pathlib import Path

from pathoinfer._extras import extra, require
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Runtime


def load_plan(card: ModelCard, folder: Path) -> Runtime:
    require("tensorrt", extra("trt"))
    raise NotImplementedError("lands in v0.2")
