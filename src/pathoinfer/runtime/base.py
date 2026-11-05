"""Runtime protocol: one batch of tiles in, raw model outputs out. Knows nothing about slides.

Every runtime calls :func:`preprocess`, driven only by the ModelCard, so local and remote
runtimes feed a model exactly what training did. Every runtime runs fixed-size batches
(:func:`run_in_fixed_batches`): the last batch is padded with zero tiles and the padding
stripped from the outputs, so a backend only ever sees one input shape.
"""

from collections.abc import Callable
from typing import Protocol

import torch

from pathoinfer.registry.model_card import ModelCard

Outputs = dict[str, torch.Tensor]

DTYPES = {"float32": torch.float32, "float16": torch.float16, "uint8": torch.uint8}


class Runtime(Protocol):
    card: ModelCard
    device: torch.device
    batch_size: int

    def predict(self, batch: torch.Tensor) -> Outputs:
        """Run ``(N, H, W, 3)`` RGB uint8 tiles (CPU, pinned or on ``device``); return the
        model's outputs keyed by ``card.output.tensor_names()``, ``N`` rows each, on
        ``device``, as the model produced them (no activation or post-processing)."""
        ...

    def warmup(self) -> None:
        """Run one zero batch so the first real batch pays no load or compile cost."""
        ...


def resolve_device(device: str) -> torch.device:
    """``auto`` (CUDA when available, else CPU), ``cpu``, ``cuda`` or ``cuda:N``."""
    if device == "auto":
        return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", 0)
    return resolved


def preprocess(batch: torch.Tensor, card: ModelCard, device: torch.device) -> torch.Tensor:
    """``(N, H, W, 3)`` RGB uint8 tiles to the card's model input on ``device``.

    The uint8 batch moves to ``device`` first (4x fewer bytes than float). Then, as
    ``card.input`` says: reorder channels, ``(x * scale - mean) / std`` per channel in that
    order (skipped for a ``uint8`` input, which normalises inside the model), lay out as
    NCHW or NHWC and cast to the input dtype.
    """
    spec = card.input
    x = batch.to(device, non_blocking=True)
    if spec.channel_order == "BGR":
        x = x.flip(-1)
    if spec.dtype != "uint8":
        norm = spec.normalisation
        mean = torch.tensor(norm.mean, dtype=torch.float32, device=device)
        std = torch.tensor(norm.std, dtype=torch.float32, device=device)
        x = (x.float() * norm.scale - mean) / std
    if spec.layout == "NCHW":
        x = x.permute(0, 3, 1, 2)
    return x.to(DTYPES[spec.dtype]).contiguous()


def run_in_fixed_batches(
    batch: torch.Tensor, batch_size: int, run: Callable[[torch.Tensor], Outputs]
) -> Outputs:
    """Call ``run`` on ``batch_size`` tiles at a time, zero-padding the last chunk, and
    return the outputs of the real tiles only, concatenated."""
    if batch.shape[0] == 0:
        raise ValueError("empty batch")
    parts: list[Outputs] = []
    for start in range(0, batch.shape[0], batch_size):
        chunk = batch[start : start + batch_size]
        real = chunk.shape[0]
        if real < batch_size:
            padding = chunk.new_zeros((batch_size - real, *chunk.shape[1:]))
            chunk = torch.cat([chunk, padding])
        parts.append({name: out[:real] for name, out in run(chunk).items()})
    if len(parts) == 1:
        return parts[0]
    return {name: torch.cat([part[name] for part in parts]) for name in parts[0]}
