"""Stitcher protocol: tile outputs in, one slide-level result out. Lands in v0.1."""

from typing import Protocol

from pathoinfer.runtime.base import Outputs
from pathoinfer.tiling.grid import TileSpec


class Stitcher[R](Protocol):
    """Fed from one thread, in any tile order; ``finish`` is called once, after the last tile."""

    def add(self, tile: TileSpec, outputs: Outputs) -> None:
        """Merge one tile's outputs (one batch row) into the slide-level state."""
        ...

    def finish(self) -> R:
        """Return the slide-level result, in level-0 pixel coordinates."""
        ...
