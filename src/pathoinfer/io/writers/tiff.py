"""Pyramidal TIFFs of rasters (label maps, heatmaps) that slide readers open.

A generic tiled pyramid, as OpenSlide reads it ("generic-tiff"), and so fastslide and
QuPath: every level is its own top-level image, full resolution first, each next one
halved until it fits one tile (floats area-averaged, integers by nearest neighbour).
Levels stored as SubIFDs are invisible to OpenSlide, which then sees one level. Tiles
are 256 px, zlib-compressed. Resolution tags give each level's pixel size, which the
readers report as mpp; the first level's ImageDescription holds the level-0 transform
and the run metadata as JSON.

Integer rasters (label maps) are written as 8-bit: OpenSlide decodes through 8-bit
RGBA and reads 16-bit as zeros. Class ids stay as they are, and ``NODATA`` (255) marks
pixels no tile reached (-1 in the raster). Float rasters stay float32, which tifffile
and Bio-Formats read but OpenSlide and fastslide do not.
"""

import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
from loguru import logger
from numpy.typing import NDArray

from pathoinfer._logging import log_call
from pathoinfer.io.objects import RunInfo
from pathoinfer.stitch.base import Raster

TILE = 256
NODATA = 255


@log_call
def write_tiff(path: str | Path, raster: Raster, run: RunInfo) -> Path:
    """Write a 2D ``raster`` whose origin is the slide's level-0 origin."""
    path = Path(path)
    if raster.array.ndim != 2:
        raise ValueError(f"expected a 2D raster, got shape {raster.array.shape}")
    # shortcut: no TIFF position tags; add XPosition/YPosition if rasters gain offsets.
    if raster.origin != (0.0, 0.0):
        raise ValueError(f"raster origin {raster.origin} is not the slide's (0, 0)")
    levels = _pyramid(_storable(raster.array))
    um_per_px = run.slide_mpp * raster.scale
    description = json.dumps(
        {"level0_per_px": raster.scale, "um_per_px": um_per_px, "run": run.to_dict()}
    )
    partial = path.with_name(path.name + ".partial")
    with tifffile.TiffWriter(partial, bigtiff=True) as tif:
        for i, level in enumerate(levels):
            px_per_cm = 1e4 / um_per_px * level.shape[1] / raster.array.shape[1]
            tif.write(
                level,
                tile=(TILE, TILE),
                compression="zlib",
                photometric="minisblack",
                resolution=(px_per_cm, px_per_cm),
                resolutionunit="CENTIMETER",
                subfiletype=1 if i else 0,  # reduced-resolution image
                description=description if i == 0 else None,
                metadata=None,
            )
    os.replace(partial, path)
    logger.info("wrote {} ({} levels)", path, len(levels))
    return path


def _storable(array: NDArray[Any]) -> NDArray[Any]:
    """Integer rasters as uint8 (-1 -> ``NODATA``); floats as float32."""
    if not np.issubdtype(array.dtype, np.integer):
        return array.astype(np.float32, copy=False)
    if array.size and (array.min() < -1 or array.max() >= NODATA):
        raise ValueError(f"label values must be -1 to {NODATA - 1} to fit 8 bits")
    return np.where(array < 0, NODATA, array).astype(np.uint8)


def _pyramid(array: NDArray[Any]) -> list[NDArray[Any]]:
    integer = np.issubdtype(array.dtype, np.integer)
    levels = [array]
    while max(levels[-1].shape) > TILE:
        h, w = levels[-1].shape
        size = (max(1, w // 2), max(1, h // 2))
        interpolation = cv2.INTER_NEAREST if integer else cv2.INTER_AREA
        levels.append(np.asarray(cv2.resize(levels[-1], size, interpolation=interpolation)))
    return levels
