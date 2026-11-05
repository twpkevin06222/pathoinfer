from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

WIDTH, HEIGHT = 4096, 3072
SLIDE_MPP = 0.25


def write_pyramid(path: Path, mpp: float | None, levels: int = 4) -> Path:
    """A tiled pyramidal TIFF (downsamples 1, 2, 4, ...) with two blobs of pink tissue on
    near-white glass, which OpenSlide reads as a generic TIFF."""
    image = np.full((HEIGHT, WIDTH, 3), 240, np.uint8)
    cv2.ellipse(image, (1500, 1400), (900, 700), 0, 0, 360, (190, 110, 160), -1)
    cv2.rectangle(image, (2900, 400), (3700, 1200), (150, 80, 140), -1)
    noise = np.random.default_rng(0).integers(-12, 12, image.shape)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    options: dict[str, object] = {"tile": (256, 256), "photometric": "rgb", "compression": "zlib"}
    if mpp is not None:
        options |= {"resolution": (1e4 / mpp, 1e4 / mpp), "resolutionunit": "CENTIMETER"}
    with tifffile.TiffWriter(path) as tif:
        for level in range(levels):
            tif.write(image, subfiletype=0 if level == 0 else 1, **options)
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), cv2.INTER_AREA)
    return path


@pytest.fixture(scope="session")
def slide_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_pyramid(tmp_path_factory.mktemp("slides") / "synthetic.tiff", SLIDE_MPP)


@pytest.fixture(scope="session")
def slide_without_mpp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return write_pyramid(tmp_path_factory.mktemp("slides") / "no_mpp.tiff", None)
