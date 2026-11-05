"""TorchScript or torch.export models under inference_mode, with an optional autocast. torch is
not a pathoinfer extra: install the build that matches your CUDA. Lands in v0.1.
"""

from pathlib import Path

from pathoinfer._extras import require
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Runtime


def load(card: ModelCard, folder: Path, autocast: bool = False) -> Runtime:
    require("torch", "install torch for your CUDA version (https://pytorch.org)")
    raise NotImplementedError("lands in v0.1")
