"""Thin wrapper over patchology: tissue mask, overlapping square grid, and a TileSpec per
tile with its core rectangle. Slides are read through patchology only; pathoinfer has no
reader backends of its own. Lands in v0.1."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TileSpec:
    """One tile, in level-0 pixels (x right, y down).

    ``x, y, size`` is the region read and fed to the model. ``core`` is the
    ``(x0, y0, x1, y1)`` part this tile owns when stitching: overlapping neighbours
    split their overlap, so every level-0 pixel belongs to exactly one core.
    """

    x: int
    y: int
    size: float
    core: tuple[int, int, int, int]
