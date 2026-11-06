"""make_runtime: the runtime for a model folder, chosen by the card's ``framework``."""

from collections.abc import Callable
from pathlib import Path

from pathoinfer._logging import log_call
from pathoinfer.config import RuntimeConfig
from pathoinfer.registry.model_card import Framework, ModelCard
from pathoinfer.runtime.base import Runtime
from pathoinfer.runtime.local.onnx import OnnxRuntime
from pathoinfer.runtime.local.tensorrt import TensorRTRuntime
from pathoinfer.runtime.local.torch import TorchRuntime


def _onnx(card: ModelCard, path: Path, config: RuntimeConfig) -> Runtime:
    return OnnxRuntime(
        card, path, device=config.device, batch_size=config.batch_size, tensorrt=config.tensorrt
    )


def _torch(card: ModelCard, path: Path, config: RuntimeConfig) -> Runtime:
    return TorchRuntime(
        card, path, device=config.device, batch_size=config.batch_size, autocast=config.autocast
    )


def _tensorrt(card: ModelCard, path: Path, config: RuntimeConfig) -> Runtime:
    return TensorRTRuntime(card, path, device=config.device, batch_size=config.batch_size)


def _safetensors(card: ModelCard, path: Path, config: RuntimeConfig) -> Runtime:
    raise ValueError(f"{path} holds weights only; export the model to ONNX or TorchScript")


BUILDERS: dict[Framework, Callable[[ModelCard, Path, RuntimeConfig], Runtime]] = {
    "onnx": _onnx,
    "torch": _torch,
    "tensorrt": _tensorrt,
    "safetensors": _safetensors,
}


@log_call
def make_runtime(card: ModelCard, model_dir: Path, config: RuntimeConfig) -> Runtime:
    """``model_dir`` is the folder ``registry.loader.load_model`` returned with ``card``."""
    return BUILDERS[card.framework](card, model_dir / card.file, config)
