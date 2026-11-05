"""Runtime protocol: one batch of tiles in, raw model outputs out. Knows nothing about slides.

Preprocessing lives here, driven by the ModelCard, so every runtime (local or remote)
normalises tiles exactly as training did: the single source of training/inference
parity. Lands in v0.1.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from pathoinfer.registry.model_card import ModelCard

Outputs = dict[str, NDArray[np.generic]]


class Runtime(Protocol):
    card: ModelCard

    def predict(self, batch: NDArray[np.uint8]) -> Outputs:
        """Run ``batch`` (N, H, W, 3) RGB uint8 tiles; return outputs keyed by
        ``card.outputs`` names, as the model produced them (no post-processing)."""
        ...

    def warmup(self) -> None:
        """Run one dummy batch so the first real batch pays no compile or load cost."""
        ...


def preprocess(batch: NDArray[np.uint8], card: ModelCard) -> NDArray[np.float32]:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float32: ``x / 255``, then ``(x - mean) / std``."""
    raise NotImplementedError("lands in v0.1")
