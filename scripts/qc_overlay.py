"""Overlay a GeoJSON output on its slide to check alignment by eye::

    uv run python scripts/qc_overlay.py slide_name-01.svs outputs/slide_name-01.geojson.gz

Writes one PNG (default: next to the GeoJSON, ``<name>.qc.png``): the slide thumbnail
with every object drawn in its class colour, and three crops at high zoom (level 0 by
default) centred on random points of object outlines, where misalignment shows first.
The crops are marked 1-3 on the thumbnail. Outlines are drawn on pixel edges, as the
coordinates mean (level-0 pixel ``(i, j)`` covers ``[i, i+1] x [j, j+1]``).
"""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import shapely
from loguru import logger

from patchology import open_slide

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SHIFT = 4  # cv2 sub-pixel bits


def load(path: Path) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rt") as f:
        features = json.load(f)["features"]
    geometries = np.array([shapely.from_geojson(json.dumps(f["geometry"])) for f in features])
    colors = [tuple(f["properties"]["classification"]["color"]) for f in features]
    return geometries, colors


def draw(
    image: np.ndarray,
    geometries: np.ndarray,
    colors: list[Any],
    origin: tuple[float, float],
    level0_per_px: float,
    width: int,
) -> None:
    """Draw ``geometries`` (level-0 px) on ``image``, whose pixel (0, 0) starts at
    ``origin`` and covers ``level0_per_px`` level-0 px."""
    scale = 2**SHIFT / level0_per_px
    for geometry, color in zip(geometries, colors, strict=True):
        if geometry.geom_type == "Point":
            xy = (np.array([geometry.x, geometry.y]) - origin) / level0_per_px - 0.5
            centre = tuple(int(v) for v in np.round(xy * 2**SHIFT))
            cv2.circle(image, centre, 3 * 2**SHIFT, color, width, cv2.LINE_AA, SHIFT)
            continue
        rings = [
            r for p in getattr(geometry, "geoms", [geometry]) for r in (p.exterior, *p.interiors)
        ]
        lines = [
            np.round((np.asarray(r.coords) - origin) * scale - 2 ** (SHIFT - 1)).astype(np.int32)
            for r in rings
        ]
        cv2.polylines(image, lines, True, color, width, cv2.LINE_AA, SHIFT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("slide", type=Path)
    parser.add_argument("geojson", type=Path)
    parser.add_argument("--out", type=Path, help="PNG to write (default: <geojson>.qc.png)")
    parser.add_argument("--thumbnail", type=int, default=2048, help="thumbnail size, px")
    parser.add_argument("--crop", type=int, default=512, help="crop size, px")
    parser.add_argument(
        "--downsample", type=float, default=1.0, help="crop zoom, level-0 px per px"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out or args.geojson.with_name(args.geojson.name.split(".geojson")[0] + ".qc.png")

    geometries, colors = load(args.geojson)
    if len(geometries) == 0:
        raise SystemExit(f"{args.geojson} has no objects")
    tree = shapely.STRtree(geometries)
    rng = np.random.default_rng(args.seed)
    picks = rng.choice(len(geometries), size=min(3, len(geometries)), replace=False)
    side = args.crop * args.downsample

    with open_slide(args.slide) as slide:
        thumbnail = np.array(slide.get_thumbnail((args.thumbnail, args.thumbnail)))
        thumb_scale = slide.dimensions[0] / thumbnail.shape[1]
        draw(thumbnail, geometries, colors, (0, 0), thumb_scale, 1)
        crops = []
        for n, i in enumerate(picks, 1):
            boundary = (
                geometries[i].boundary if geometries[i].geom_type != "Point" else geometries[i]
            )
            point = (
                shapely.line_interpolate_point(boundary, rng.random(), normalized=True)
                if boundary.geom_type != "Point"
                else boundary
            )
            x0, y0 = int(point.x - side / 2), int(point.y - side / 2)
            crop = slide.read_region_scaled((x0, y0), args.downsample, (args.crop, args.crop))
            near = tree.query(shapely.box(x0, y0, x0 + side, y0 + side))
            draw(crop, geometries[near], [colors[j] for j in near], (x0, y0), args.downsample, 2)
            crops.append((crop, (x0, y0)))
            corner = np.array([x0, y0]) / thumb_scale
            size = side / thumb_scale
            cv2.rectangle(
                thumbnail,
                tuple(corner.astype(int)),
                tuple((corner + size).astype(int)),
                (0, 0, 0),
                2,
            )
            cv2.putText(
                thumbnail,
                str(n),
                tuple((corner + [size + 4, 0]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                2,
            )

    fig = plt.figure(figsize=(16, 10), layout="constrained")
    grid = fig.add_gridspec(3, 2, width_ratios=[2, 1])
    ax = fig.add_subplot(grid[:, 0])
    ax.imshow(thumbnail)
    ax.set_title(f"{args.slide.name}: {len(geometries)} objects ({args.geojson.name})")
    ax.axis("off")
    for n, (crop, (x0, y0)) in enumerate(crops):
        ax = fig.add_subplot(grid[n, 1])
        ax.imshow(crop)
        ax.set_title(f"{n + 1}: level-0 ({x0}, {y0}), {args.downsample:g} px/px", fontsize=9)
        ax.axis("off")
    fig.savefig(out, dpi=100)
    logger.info("wrote {}", out)


if __name__ == "__main__":
    main()
