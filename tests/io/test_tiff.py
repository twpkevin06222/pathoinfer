import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from patchology import open_slide
from pathoinfer.io.objects import RunInfo
from pathoinfer.io.writers.tiff import NODATA, write_tiff
from pathoinfer.stitch.base import Raster


def labels(shape: tuple[int, int] = (700, 1100)) -> np.ndarray:
    """Class ids 0-6, and -1 (no tile reached) in a corner."""
    rng = np.random.default_rng(0)
    array = rng.integers(0, 7, shape).astype(np.int16)
    array[:40, :40] = -1
    return array


def test_levels_are_top_level_images_with_resolution(tmp_path: Path, seg_run: RunInfo) -> None:
    path = write_tiff(tmp_path / "labels.tiff", Raster(labels(), scale=4.0), seg_run)
    with tifffile.TiffFile(path) as tif:
        pages = list(tif.pages)  # no SubIFDs: OpenSlide only sees top-level images
        assert [p.shape for p in pages] == [(700, 1100), (350, 550), (175, 275), (87, 137)]
        assert all(p.is_tiled and p.tilewidth == 256 for p in pages)
        assert all(p.dtype == np.uint8 for p in pages)
        assert pages[0].compression == tifffile.COMPRESSION.ADOBE_DEFLATE
        um_per_px = seg_run.slide_mpp * 4.0
        assert pages[0].get_resolution(tifffile.RESUNIT.MICROMETER) == pytest.approx(
            (1 / um_per_px, 1 / um_per_px)
        )
        assert pages[1].get_resolution(tifffile.RESUNIT.MICROMETER)[0] == pytest.approx(
            0.5 / um_per_px
        )
        description = json.loads(pages[0].description)
        assert description["level0_per_px"] == 4.0
        assert description["run"]["slide_id"] == seg_run.slide_id


@pytest.mark.parametrize("backend", ["openslide", "fastslide"])
def test_slide_readers_open_the_pyramid(tmp_path: Path, seg_run: RunInfo, backend: str) -> None:
    """OpenSlide and fastslide (and QuPath, through OpenSlide) see every level, the pixel
    size, and the class ids exactly."""
    array = labels()
    path = write_tiff(tmp_path / "labels.tiff", Raster(array, scale=4.0), seg_run)
    with open_slide(path, backend) as slide:
        assert slide.level_count == 4
        assert slide.dimensions == (1100, 700)
        assert slide.mpp == pytest.approx((seg_run.slide_mpp * 4.0,) * 2)
        read = np.asarray(slide.read_region((0, 0), 0, (1100, 700)))[..., 0]
    np.testing.assert_array_equal(read, np.where(array < 0, NODATA, array))


def test_float_rasters_stay_float(tmp_path: Path, seg_run: RunInfo) -> None:
    array = np.random.default_rng(0).uniform(0, 1, (300, 400)).astype(np.float32)
    path = write_tiff(tmp_path / "heat.tiff", Raster(array, scale=4.0), seg_run)
    with tifffile.TiffFile(path) as tif:
        np.testing.assert_array_equal(tif.pages[0].asarray(), array)


def test_labels_that_do_not_fit_8_bits_are_refused(tmp_path: Path, seg_run: RunInfo) -> None:
    with pytest.raises(ValueError, match="8 bits"):
        write_tiff(tmp_path / "x.tiff", Raster(np.full((4, 4), 300, np.int16), 1.0), seg_run)


def test_offset_rasters_are_refused(tmp_path: Path, seg_run: RunInfo) -> None:
    with pytest.raises(ValueError, match="origin"):
        write_tiff(tmp_path / "x.tiff", Raster(np.zeros((4, 4)), 1.0, (8.0, 0.0)), seg_run)
