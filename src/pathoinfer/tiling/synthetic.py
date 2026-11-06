"""A synthetic slide for tests, demos and the Docker smoke run: no real slide needed."""

from pathlib import Path

import cv2
import numpy as np
import tifffile

WIDTH, HEIGHT = 4096, 3072
SLIDE_MPP = 0.25


def write_pyramid(path: Path, mpp: float | None = SLIDE_MPP, levels: int = 4) -> Path:
    """A tiled pyramidal TIFF (downsamples 1, 2, 4, ...) with two blobs of pink tissue on
    near-white glass, which OpenSlide reads as a generic TIFF. ``mpp`` None writes no
    resolution, so pathoinfer refuses the slide."""
    image = np.full((HEIGHT, WIDTH, 3), 240, np.uint8)
    cv2.ellipse(image, (1500, 1400), (900, 700), 0, 0, 360, (190, 110, 160), -1)
    cv2.rectangle(image, (2900, 400), (3700, 1200), (150, 80, 140), -1)
    noise = np.random.default_rng(0).integers(-12, 12, image.shape)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    resolution = None if mpp is None else (1e4 / mpp, 1e4 / mpp)
    with tifffile.TiffWriter(path) as tif:
        for level in range(levels):
            tif.write(
                image,
                subfiletype=0 if level == 0 else 1,
                tile=(256, 256),
                photometric="rgb",
                compression="zlib",
                resolution=resolution,
                resolutionunit="CENTIMETER" if resolution else None,
            )
            half = (image.shape[1] // 2, image.shape[0] // 2)
            image = np.asarray(cv2.resize(image, half, interpolation=cv2.INTER_AREA))
    return path
