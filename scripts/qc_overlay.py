"""Overlay a GeoJSON output on its whole slide, to check it by eye::

    uv run python scripts/qc_overlay.py slide_name-01.svs outputs/slide_name-01.geojson.gz

Writes one PNG (default: next to the GeoJSON, ``<name>.qc.png``) of the whole slide read
at ``--mpp`` (default 6 µm per px), drawn by ``pathoinfer.io.writers.overlay.render``:
every object filled in its class colour at ``--alpha`` opacity and outlined, with a
legend. Segmentation runs write the same picture themselves (``outputs.overlay``); this
script draws any GeoJSON output, detections included.
"""

import argparse
import gzip
import json
from pathlib import Path

import cv2
import numpy as np
import shapely
from loguru import logger

from patchology import open_slide
from pathoinfer.io.writers.overlay import ALPHA, render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("slide", type=Path)
    parser.add_argument("geojson", type=Path)
    parser.add_argument("--out", type=Path, help="PNG to write (default: <geojson>.qc.png)")
    parser.add_argument("--mpp", type=float, default=6.0, help="image resolution, µm per px")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="fill opacity, 0 to 1")
    args = parser.parse_args()
    out = args.out or args.geojson.with_name(args.geojson.name.split(".geojson")[0] + ".qc.png")

    opener = gzip.open if args.geojson.name.endswith(".gz") else open
    with opener(args.geojson, "rt") as f:
        features = json.load(f)["features"]
    if not features:
        raise SystemExit(f"{args.geojson} has no objects")
    geometries = [shapely.from_geojson(json.dumps(f["geometry"])) for f in features]
    classes = [f["properties"]["classification"] for f in features]
    colors = [tuple(c["color"]) for c in classes]
    legend = {c["name"]: tuple(c["color"]) for c in classes}

    with open_slide(args.slide) as slide:
        if slide.mpp is None:
            raise SystemExit(f"{args.slide} has no resolution; cannot read it at {args.mpp} mpp")
        level0_per_px = args.mpp / slide.mpp[0]
        size = tuple(max(1, int(d / level0_per_px)) for d in slide.dimensions)
        image = slide.read_region_scaled((0, 0), level0_per_px, size)
    drawn = render(np.asarray(image), geometries, colors, legend, level0_per_px, args.alpha)
    cv2.imwrite(str(out), cv2.cvtColor(drawn, cv2.COLOR_RGB2BGR))
    logger.info(
        "wrote {} ({} x {} px at {:g} mpp, {} objects)", out, *size, args.mpp, len(features)
    )


if __name__ == "__main__":
    main()
