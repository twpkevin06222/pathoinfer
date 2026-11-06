"""TensorRT plan runtime: runs a plan built by ``convert.tensorrt`` for this GPU.

Inputs and outputs stay on the GPU as torch tensors; their addresses are handed to the
execution context, so there is no host copy. The plan's optimisation profile covers the
runtime's ``batch_size``; batches are padded to it like every other runtime.
"""

from pathlib import Path
from typing import Any

import torch

from pathoinfer._extras import extra, require
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Outputs, preprocess, resolve_device, run_in_fixed_batches


class TensorRTRuntime:
    def __init__(self, card: ModelCard, plan: Path, device: str = "auto", batch_size: int = 32):
        trt = require("tensorrt", extra("trt"))
        self.device = resolve_device(device)
        if self.device.type != "cuda":
            raise ValueError("TensorRT plans need a CUDA device")
        self.card, self.batch_size = card, batch_size
        self._trt = trt
        logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self.device):
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(plan.read_bytes())
            if self.engine is None:
                raise ValueError(f"{plan}: not a TensorRT plan for this GPU and TensorRT version")
            self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self._input = next(
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        self._outputs = card.output.tensor_names()

    def predict(self, batch: torch.Tensor) -> Outputs:
        return run_in_fixed_batches(batch, self.batch_size, self._run)

    def warmup(self) -> None:
        size = self.card.input.tile_size
        self.predict(torch.zeros((self.batch_size, size, size, 3), dtype=torch.uint8))

    def _run(self, chunk: torch.Tensor) -> Outputs:
        x = preprocess(chunk, self.card, self.device).contiguous()
        x = x.to(self._dtype(self._input))
        self.context.set_input_shape(self._input, tuple(x.shape))
        self.context.set_tensor_address(self._input, x.data_ptr())
        outputs = {}
        for name in self._outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            outputs[name] = torch.empty(shape, dtype=self._dtype(name), device=self.device)
            self.context.set_tensor_address(name, outputs[name].data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        stream.synchronize()
        return outputs

    def _dtype(self, name: str) -> Any:
        trt = self._trt
        return {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.bfloat16: torch.bfloat16,
            trt.int32: torch.int32,
            trt.int64: torch.int64,
            trt.bool: torch.bool,
        }[self.engine.get_tensor_dtype(name)]
