"""TorchScript or torch.export models under inference_mode, with an optional autocast.
torch is imported on first use, so ``import pathoinfer`` stays light. A model whose graph
calls torchvision ops (e.g. ``torchvision::nms``) also needs torchvision installed and
imported before loading. Lands in v0.1.
"""

from pathlib import Path

from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Runtime


def load(card: ModelCard, folder: Path, autocast: bool = False) -> Runtime:
    raise NotImplementedError("lands in v0.1")
