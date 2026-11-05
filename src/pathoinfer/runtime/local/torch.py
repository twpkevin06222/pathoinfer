"""PyTorch runtime: TorchScript (``.pt``, ``.pth``, ``.ts``) or ``torch.export`` (``.pt2``)
models under ``inference_mode``, with optional bf16 / fp16 autocast. A model whose graph
calls torchvision ops (e.g. ``torchvision::nms``) also needs torchvision imported first.
"""

from pathlib import Path
from typing import Literal

import torch

from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Outputs, preprocess, resolve_device, run_in_fixed_batches

Autocast = Literal["bf16", "fp16"]
AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


class TorchRuntime:
    def __init__(
        self,
        card: ModelCard,
        model_path: Path,
        *,
        device: str = "auto",
        batch_size: int = 32,
        autocast: Autocast | None = None,
    ) -> None:
        self.card, self.batch_size, self.autocast = card, batch_size, autocast
        self.device = resolve_device(device)
        if model_path.suffix == ".pt2":
            module = torch.export.load(str(model_path)).module().to(self.device)
        else:
            jit = torch.jit.load(str(model_path), map_location=self.device)  # type: ignore[no-untyped-call]
            module = jit.eval()
        self.module = module
        self._outputs = card.output.tensor_names()

    def predict(self, batch: torch.Tensor) -> Outputs:
        return run_in_fixed_batches(batch, self.batch_size, self._run)

    def warmup(self) -> None:
        size = self.card.input.tile_size
        self.predict(torch.zeros((self.batch_size, size, size, 3), dtype=torch.uint8))

    def _run(self, chunk: torch.Tensor) -> Outputs:
        x = preprocess(chunk, self.card, self.device)
        dtype = AUTOCAST_DTYPES[self.autocast] if self.autocast else None
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=dtype, enabled=dtype is not None),
        ):
            out = self.module(x)
        if isinstance(out, torch.Tensor):
            out = (out,)
        if isinstance(out, dict):
            return {name: out[name] for name in self._outputs}
        return dict(zip(self._outputs, out, strict=True))
