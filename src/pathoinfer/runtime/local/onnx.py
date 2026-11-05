"""ONNX Runtime ([onnx] extra): TensorRT EP when its libraries load, else CUDA, else CPU.

On CUDA the input is preprocessed on the GPU and bound with IO binding through DLPack,
and the outputs stay on the GPU (``torch.from_dlpack``): no host round trip. The input
dtype comes from the card (float16 for an fp16 graph) and must match the graph's.
"""

import ctypes
import sys
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.utils.dlpack import from_dlpack

from pathoinfer._extras import extra, require
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Outputs, preprocess, resolve_device, run_in_fixed_batches

TRT_CACHE_DIR = Path.home() / ".cache" / "pathoinfer" / "trt"
GRAPH_DTYPES = {"tensor(float)": "float32", "tensor(float16)": "float16", "tensor(uint8)": "uint8"}
# shortcut: the TensorRT major onnxruntime-gpu 1.31 links against; update with onnxruntime.
NVINFER = "libnvinfer.so.10"


class OnnxRuntime:
    def __init__(
        self,
        card: ModelCard,
        model_path: Path,
        *,
        device: str = "auto",
        batch_size: int = 32,
        tensorrt: bool = True,
        trt_cache_dir: Path = TRT_CACHE_DIR,
    ) -> None:
        self._ort = require("onnxruntime", extra("onnx"))
        self.card, self.batch_size = card, batch_size
        wanted = resolve_device(device)
        providers = _providers(self._ort, wanted, tensorrt, trt_cache_dir, card.precision)
        self.session = self._ort.InferenceSession(str(model_path), providers=providers)
        on_gpu = "CUDAExecutionProvider" in self.session.get_providers()
        if wanted.type == "cuda" and not on_gpu:
            if device != "auto":
                raise RuntimeError(f"{device} requested, but onnxruntime cannot use CUDA here")
            wanted = torch.device("cpu")
        self.device = wanted
        self.providers: list[str] = self.session.get_providers()
        self._outputs = card.output.tensor_names()
        _check_signature(self.session, card, model_path)

    def predict(self, batch: torch.Tensor) -> Outputs:
        return run_in_fixed_batches(batch, self.batch_size, self._run)

    def warmup(self) -> None:
        size = self.card.input.tile_size
        self.predict(torch.zeros((self.batch_size, size, size, 3), dtype=torch.uint8))

    def _run(self, chunk: torch.Tensor) -> Outputs:
        x = preprocess(chunk, self.card, self.device)
        if self.device.type == "cpu":
            arrays = self.session.run(list(self._outputs), {self.card.input.name: x.numpy()})
            return {
                name: torch.from_numpy(a) for name, a in zip(self._outputs, arrays, strict=True)
            }
        binding = self.session.io_binding()
        binding.bind_ortvalue_input(self.card.input.name, self._ort.OrtValue.from_dlpack(x))
        for name in self._outputs:
            binding.bind_output(name, "cuda", self.device.index)
        torch.cuda.current_stream(self.device).synchronize()  # onnxruntime uses its own stream
        self.session.run_with_iobinding(binding)
        values = binding.get_outputs()
        return {name: from_dlpack(v) for name, v in zip(self._outputs, values, strict=True)}


def _providers(
    ort: Any, device: torch.device, tensorrt: bool, cache_dir: Path, precision: str
) -> list[Any]:
    if device.type == "cpu":
        return ["CPUExecutionProvider"]
    available = ort.get_available_providers()
    providers: list[Any] = []
    if tensorrt and "TensorrtExecutionProvider" in available and _nvinfer_loads():
        cache_dir.mkdir(parents=True, exist_ok=True)
        options = {
            "device_id": device.index,
            "trt_fp16_enable": precision == "fp16",
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache_dir),
        }
        providers.append(("TensorrtExecutionProvider", options))
    if "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": device.index}))
    return [*providers, "CPUExecutionProvider"]


def _nvinfer_loads() -> bool:
    """Whether TensorRT's runtime library loads, so onnxruntime's TensorRT EP can start
    instead of logging an error and falling back."""
    if not sys.platform.startswith("linux"):
        return True
    try:
        ctypes.CDLL(NVINFER)
    except OSError:
        logger.debug("{} not found; TensorRT EP skipped", NVINFER)
        return False
    return True


def _check_signature(session: Any, card: ModelCard, path: Path) -> None:
    inputs = {i.name: i.type for i in session.get_inputs()}
    graph = GRAPH_DTYPES.get(inputs.get(card.input.name, ""), inputs.get(card.input.name))
    if graph != card.input.dtype:
        raise ValueError(
            f"{path}: card input {card.input.name!r} is {card.input.dtype}, graph has {graph}"
        )
    missing = set(card.output.tensor_names()) - {o.name for o in session.get_outputs()}
    if missing:
        raise ValueError(f"{path}: outputs {sorted(missing)} are not in the graph")
