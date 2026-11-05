"""Overlapping tile grid at the model's resolution, on top of patchology. Lands in v0.1.

Coordinates. Level 0 is the slide's full resolution, ``slide_mpp`` microns per pixel.
Model space is the slide resampled to the model's ``mpp``. One model pixel spans

    downsample = mpp / slide_mpp            level-0 pixels,

so ``level0 = model * downsample`` and ``model = level0 / downsample``.

Reading. A tile is ``tile_size`` model pixels square, ``tile_size * downsample`` level-0
pixels. patchology reads the highest-resolution pyramid level whose downsample ``ds_L``
is at most ``downsample`` (1% tolerance, so 4.0002 counts as 4), reads
``round(tile_size * read_factor)`` pixels there with ``read_factor = downsample / ds_L``,
and resizes them to ``tile_size`` (INTER_AREA when shrinking). Example: a 0.25 mpp slide
with levels 1, 4, 16 and a 0.5 mpp model gives downsample 2, level 0 and read factor 2;
a 1.0 mpp model gives downsample 4, level 1 and read factor 1.

Grid. Tiles start every ``stride = tile_size - overlap`` model pixels from the slide's
top-left corner, and enough columns and rows are added to cover the whole slide (edge
tiles may run past it; patchology pads with the background colour). A tile is kept when
the tissue mask touches it, or covers at least ``min_tissue_fraction`` of it.

Cores. Each tile owns its core: the tile shrunk by half the overlap on every side that has
a grid neighbour (``overlap // 2`` on the left/top, the rest on the right/bottom), and its
full extent on the slide's edges. Neighbours split their shared overlap at the same model
pixel, so the cores of all grid tiles partition the slide exactly, in model space and,
after rounding the shared boundaries once, in level 0. Kept tiles' cores never overlap;
dropped tiles hold no tissue (at the default ``min_tissue_fraction`` of 0).
"""

import math
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from patchology import (
    ArtifactConfig,
    ColorConfig,
    SlideMask,
    SlideReader,
    TilingConfig,
    TilingPipeline,
    TissueConfig,
)
from patchology.reader.base import LEVEL_DOWNSAMPLE_TOLERANCE
from patchology.tiling import tile_fractions
from patchology.tissue.mask import MaskSource
from pathoinfer.tiling.slide_id import slide_id


class SlideResolutionError(ValueError):
    """The slide does not report its resolution and no ``mpp_override`` was given."""


@dataclass(frozen=True)
class Box:
    """Half-open rectangle ``[x0, x1) x [y0, y1)``."""

    x0: int
    y0: int
    x1: int
    y1: int


@dataclass(frozen=True)
class TileSpec:
    """One tile. ``x, y`` is its level-0 top-left corner and ``size`` its level-0 side;
    ``xm, ym`` its top-left corner in model space. ``core`` (level 0, clipped to the
    slide) and ``core_model`` (model space) are the part this tile owns when stitching."""

    row: int
    col: int
    x: int
    y: int
    size: float
    xm: int
    ym: int
    core: Box
    core_model: Box


@dataclass(frozen=True)
class SlideGrid:
    slide_path: Path
    slide_id: str
    backend: str
    slide_dimensions: tuple[int, int]
    slide_mpp: float
    mpp: float
    tile_size: int
    overlap: int
    icc: bool
    downsample: float
    level: int
    read_factor: float
    n_rows: int
    n_cols: int
    tiles: tuple[TileSpec, ...]
    tissue_mask: SlideMask = field(repr=False)

    def spec(self, row: int, col: int) -> TileSpec:
        """The tile at ``(row, col)``, kept or not."""
        width, height = self.slide_dimensions
        xm, cx0, cx1 = _axis(col, self.n_cols, self.tile_size, self.overlap)
        ym, cy0, cy1 = _axis(row, self.n_rows, self.tile_size, self.overlap)
        last_col, last_row = col == self.n_cols - 1, row == self.n_rows - 1
        core = Box(
            round(cx0 * self.downsample),
            round(cy0 * self.downsample),
            width if last_col else min(round(cx1 * self.downsample), width),
            height if last_row else min(round(cy1 * self.downsample), height),
        )
        return TileSpec(
            row=row,
            col=col,
            x=round(xm * self.downsample),
            y=round(ym * self.downsample),
            size=self.tile_size * self.downsample,
            xm=xm,
            ym=ym,
            core=core,
            core_model=Box(cx0, cy0, cx1, cy1),
        )

    def to_level0(self, xm: float, ym: float) -> tuple[float, float]:
        return xm * self.downsample, ym * self.downsample

    def to_model(self, x: float, y: float) -> tuple[float, float]:
        return x / self.downsample, y / self.downsample


def build_grid(
    slide: SlideReader,
    tile_size: int,
    mpp: float,
    overlap: int = 0,
    *,
    min_tissue_fraction: float = 0.0,
    mpp_override: float | None = None,
    tissue: TissueConfig = TissueConfig(),
    artifacts: ArtifactConfig = ArtifactConfig(),
    icc: bool = True,
    tissue_mask: MaskSource | None = None,
) -> SlideGrid:
    """Grid of ``tile_size`` px tiles at ``mpp`` with ``overlap`` px (model space) between
    neighbours, keeping tiles that touch tissue.

    Tissue comes from patchology: detected on the thumbnail (ICC calibrated when ``icc``)
    or ``tissue_mask``, minus pen marks and blur as ``artifacts`` says.

    Raises:
        SlideResolutionError: the slide has no mpp in its metadata and no ``mpp_override``.
        ValueError: invalid geometry, or patchology rejects the slide.
    """
    if tile_size <= 0 or mpp <= 0:
        raise ValueError(f"tile_size and mpp must be positive, got {tile_size}, {mpp}")
    if not 0 <= overlap < tile_size:
        raise ValueError(f"overlap must be in [0, tile_size), got {overlap}")
    if not 0 <= min_tissue_fraction <= 1:
        raise ValueError(f"min_tissue_fraction must be in [0, 1], got {min_tissue_fraction}")
    if mpp_override is None and slide.mpp is None:
        raise SlideResolutionError(
            f"{slide.path} does not report its resolution (mpp); "
            "pass mpp_override only if you know it"
        )
    pipeline = TilingPipeline(
        tiling=TilingConfig(slide_mpp=mpp_override),
        tissue=tissue,
        artifacts=artifacts,
        color=ColorConfig(icc=icc),
    )
    slide_mpp = pipeline.slide_mpp(slide)
    _, detected, removed = pipeline.detect(slide, tissue_mask, slide_mpp)
    clean = detected - removed if removed is not None else detected

    downsample = mpp / slide_mpp
    level = slide.get_best_level_for_downsample(downsample * (1 + LEVEL_DOWNSAMPLE_TOLERANCE))
    width, height = slide.dimensions
    stride = tile_size - overlap
    n_cols = _count(width / downsample, tile_size, stride)
    n_rows = _count(height / downsample, tile_size, stride)

    rows, cols = np.divmod(np.arange(n_rows * n_cols), n_cols)
    origins = np.stack([cols, rows], axis=1) * stride * downsample
    fractions = tile_fractions(clean, np.round(origins).astype(np.int64), tile_size * downsample)
    keep: NDArray[np.bool_] = (fractions > 0) & (fractions >= min_tissue_fraction)

    grid = SlideGrid(
        slide_path=slide.path,
        slide_id=slide_id(slide.path),
        backend=slide.backend,
        slide_dimensions=(width, height),
        slide_mpp=slide_mpp,
        mpp=mpp,
        tile_size=tile_size,
        overlap=overlap,
        icc=icc,
        downsample=downsample,
        level=level,
        read_factor=downsample / slide.level_downsamples[level],
        n_rows=n_rows,
        n_cols=n_cols,
        tiles=(),
        tissue_mask=clean,
    )
    tiles = tuple(grid.spec(int(r), int(c)) for r, c in zip(rows[keep], cols[keep], strict=True))
    return replace(grid, tiles=tiles)


def read_tile(slide: SlideReader, grid: SlideGrid, spec: TileSpec) -> NDArray[np.uint8]:
    """``(tile_size, tile_size, 3)`` RGB uint8 at the model's mpp, ICC calibrated when the
    grid was built with ``icc``."""
    size = (grid.tile_size, grid.tile_size)
    tile: NDArray[np.uint8] = slide.read_region_scaled((spec.x, spec.y), grid.downsample, size)
    return slide.apply_icc(tile) if grid.icc else tile


def _count(length: float, tile_size: int, stride: int) -> int:
    """Fewest tiles, ``stride`` apart, that cover ``length``."""
    return max(1, math.ceil((length - tile_size) / stride - 1e-9) + 1)


def _axis(i: int, n: int, tile_size: int, overlap: int) -> tuple[int, int, int]:
    """Origin, core start and core end (model space) of tile ``i`` of ``n`` on one axis."""
    origin = i * (tile_size - overlap)
    start = origin + (overlap // 2 if i > 0 else 0)
    end = origin + tile_size - (overlap - overlap // 2 if i < n - 1 else 0)
    return origin, start, end
