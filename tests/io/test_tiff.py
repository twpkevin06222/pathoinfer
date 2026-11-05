import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from pathoinfer.io.objects import RunInfo
from pathoinfer.io.writers.tiff import write_tiff
from pathoinfer.stitch.base import Raster


@pytest.mark.parametrize("dtype", [np.float32, np.int16])
def test_pyramid_levels_tiles_and_resolution(tmp_path: Path, dtype: type, seg_run: RunInfo) -> None:
    array = (np.random.default_rng(0).uniform(0, 1, (700, 1100)) * 100).astype(dtype)
    path = write_tiff(tmp_path / "heat.tiff", Raster(array, scale=4.0), seg_run)
    with tifffile.TiffFile(path) as tif:
        levels = tif.series[0].levels
        assert [lvl.shape for lvl in levels] == [(700, 1100), (350, 550), (175, 275), (87, 137)]
        np.testing.assert_array_equal(levels[0].asarray(), array)
        page = tif.pages[0]
        assert page.is_tiled and page.tilewidth == 256
        assert page.compression == tifffile.COMPRESSION.ADOBE_DEFLATE
        um_per_px = seg_run.slide_mpp * 4.0
        assert page.get_resolution(tifffile.RESUNIT.MICROMETER) == pytest.approx(
            (1 / um_per_px, 1 / um_per_px)
        )
        description = json.loads(page.description)
        assert description["level0_per_px"] == 4.0
        assert description["run"]["slide_id"] == seg_run.slide_id
        assert levels[1].pages[0].get_resolution(tifffile.RESUNIT.MICROMETER)[0] == (
            pytest.approx(0.5 / um_per_px)
        )


def test_offset_rasters_are_refused(tmp_path: Path, seg_run: RunInfo) -> None:
    with pytest.raises(ValueError, match="origin"):
        write_tiff(tmp_path / "x.tiff", Raster(np.zeros((4, 4)), 1.0, (8.0, 0.0)), seg_run)
