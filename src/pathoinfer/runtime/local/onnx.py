"""ONNX Runtime ([onnx] extra): TensorRT EP when its libraries load, else CUDA, else CPU.

On CUDA the input is preprocessed on the GPU and bound with IO binding through DLPack,
and the outputs stay on the GPU (``torch.from_dlpack``): no host round trip. The input
dtype comes from the card (float16 for an fp16 graph) and must match the graph's. The
card's ``precision`` must match what the graph computes in (:func:`graph_precision`).
"""

import ctypes
import sys
from pathlib import Path
from typing import Any

import numpy as np
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
        graph = graph_precision(model_path)
        if graph != card.precision:
            raise ValueError(f"{model_path}: card precision is {card.precision}, graph is {graph}")

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


def graph_precision(path: Path) -> str:
    """What the ONNX graph at ``path`` computes in: ``fp8`` or ``int8`` if its
    Quantize/DequantizeLinear nodes use that type (``convert ptq``), else ``fp16`` or
    ``bf16`` if most weight elements are (``convert fp16``), else ``fp32``. Reads the
    graph only, not external weight files."""
    onnx = require("onnx", extra("onnx"))
    T = onnx.TensorProto
    graph = onnx.load(str(path), load_external_data=False).graph
    types = {t.name: t.data_type for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant" and node.attribute and node.attribute[0].HasField("t"):
            types[node.output[0]] = node.attribute[0].t.data_type
    quantised = {
        types.get(name)
        for node in graph.node
        if node.op_type in ("QuantizeLinear", "DequantizeLinear")
        for name in node.input[::2]  # the quantised data and its zero point
    }
    if quantised & {T.FLOAT8E4M3FN, T.FLOAT8E4M3FNUZ, T.FLOAT8E5M2, T.FLOAT8E5M2FNUZ}:
        return "fp8"
    if quantised & {T.INT8, T.UINT8}:
        return "int8"
    sizes: dict[int, int] = {}
    for t in graph.initializer:
        sizes[t.data_type] = sizes.get(t.data_type, 0) + int(np.prod(t.dims))
    floats = {"fp32": sizes.get(T.FLOAT, 0), "fp16": sizes.get(T.FLOAT16, 0)}
    floats["bf16"] = sizes.get(T.BFLOAT16, 0)
    return max(floats, key=floats.__getitem__) if any(floats.values()) else "fp32"


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
