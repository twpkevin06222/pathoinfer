"""PNG overlay: the whole slide at a chosen mpp, with every object filled in its class
colour at ``ALPHA`` opacity, outlined at full opacity, and a legend of the classes.
Detections drawn as points are circles. Outlines lie on pixel edges, as the coordinates
mean: level-0 pixel ``(i, j)`` covers ``[i, i+1] x [j, j+1]``.
"""

import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from pathoinfer._logging import log_call
from pathoinfer.io.objects import ObjectTable, RunInfo

ALPHA = 0.4  # fill opacity: the tissue stays readable under the colour
SHIFT = 4  # cv2 sub-pixel bits

Color = tuple[int, int, int]


def render(
    image: NDArray[np.uint8],
    geometries: Iterable[Any],
    colors: Sequence[Color],
    legend: dict[str, Color],
    level0_per_px: float,
    alpha: float = ALPHA,
) -> NDArray[np.uint8]:
    """``image`` (RGB, pixel ``(0, 0)`` at level-0 ``(0, 0)``, ``level0_per_px`` level-0 px
    per pixel) with ``geometries`` (level-0 px) filled at ``alpha``, outlined, and a
    legend of ``legend`` (class name to colour) in the top-left corner."""
    image = np.ascontiguousarray(image[..., :3])
    fill = image.copy()
    for geometry, color in zip(geometries, colors, strict=True):
        if geometry.geom_type != "Point":
            cv2.fillPoly(fill, _rings(geometry, level0_per_px), color, cv2.LINE_AA, SHIFT)
    out = np.asarray(cv2.addWeighted(fill, alpha, image, 1 - alpha, 0), dtype=np.uint8)
    for geometry, color in zip(geometries, colors, strict=True):
        if geometry.geom_type == "Point":
            xy = np.round((np.array([geometry.x, geometry.y]) / level0_per_px - 0.5) * 2**SHIFT)
            centre = (int(xy[0]), int(xy[1]))
            cv2.circle(out, centre, 3 * 2**SHIFT, color, 1, cv2.LINE_AA, SHIFT)
        else:
            cv2.polylines(out, _rings(geometry, level0_per_px), True, color, 1, cv2.LINE_AA, SHIFT)
    _legend(out, legend)
    return out


@log_call
def write_overlay(
    path: str | Path, slide: Any, objects: ObjectTable, run: RunInfo, mpp: float = 6.0
) -> Path:
    """Read ``slide`` (an open patchology slide) at ``mpp`` and write the overlay of
    ``objects`` as a PNG. The level-0 resolution is ``run.slide_mpp``, so an
    ``mpp_override`` applies here too."""
    path = Path(path)
    level0_per_px = mpp / run.slide_mpp
    size = tuple(max(1, int(d / level0_per_px)) for d in slide.dimensions)
    image = slide.read_region_scaled((0, 0), level0_per_px, size)
    colors: dict[int, Color] = {c.id: c.color for c in run.classes}
    present = set(objects.class_id.tolist())
    legend = {c.name: colors[c.id] for c in run.classes if c.id in present}
    drawn = render(
        image,
        objects.geometry,
        [colors[i] for i in objects.class_id.tolist()],
        legend,
        level0_per_px,
    )
    partial = path.with_name(path.name + ".partial.png")
    if not cv2.imwrite(str(partial), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR)):
        raise OSError(f"could not write {partial}")
    os.replace(partial, path)
    logger.info("wrote {} ({} x {} px at {:g} mpp)", path, *size, mpp)
    return path


def _rings(geometry: Any, level0_per_px: float) -> list[NDArray[np.int32]]:
    """``geometry``'s rings in image px, cv2 fixed point."""
    scale = 2**SHIFT / level0_per_px
    return [
        np.round(np.asarray(r.coords) * scale - 2 ** (SHIFT - 1)).astype(np.int32)
        for p in getattr(geometry, "geoms", [geometry])
        for r in (p.exterior, *p.interiors)
    ]


def _legend(image: NDArray[np.uint8], legend: dict[str, Color]) -> None:
    if not legend:
        return
    font, scale, line = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 24
    width = max(cv2.getTextSize(name, font, scale, 1)[0][0] for name in legend) + 50
    cv2.rectangle(image, (8, 8), (8 + width, 16 + line * len(legend)), (255, 255, 255), -1)
    for n, (name, color) in enumerate(sorted(legend.items())):
        y = 12 + line * n
        cv2.rectangle(image, (14, y + 4), (34, y + 18), color, -1)
        cv2.putText(image, name, (42, y + 17), font, scale, (0, 0, 0), 1, cv2.LINE_AA)
