from pathlib import Path

import numpy as np
import pytest

from patchology import open_slide
from pathoinfer.tiling.grid import SlideGrid, SlideResolutionError, build_grid, read_tile
from pathoinfer.tiling.slide_id import slide_id

from .conftest import HEIGHT, WIDTH


def grid_for(path: Path, **kwargs: object) -> SlideGrid:
    options: dict[str, object] = {"tile_size": 256, "mpp": 0.5, "overlap": 32} | kwargs
    with open_slide(path) as slide:
        return build_grid(slide, **options)  # type: ignore[arg-type]


def ownership(grid: SlideGrid, specs: list) -> np.ndarray:  # type: ignore[type-arg]
    """How many cores claim each level-0 pixel."""
    count = np.zeros((HEIGHT, WIDTH), np.uint8)
    for s in specs:
        count[s.core.y0 : s.core.y1, s.core.x0 : s.core.x1] += 1
    return count


@pytest.mark.parametrize("overlap", [0, 32, 33])
@pytest.mark.parametrize("mpp", [0.25, 0.5, 0.6, 1.0])
def test_cores_of_the_whole_grid_partition_the_slide(
    slide_path: Path, mpp: float, overlap: int
) -> None:
    grid = grid_for(slide_path, mpp=mpp, overlap=overlap)
    every = [grid.spec(r, c) for r in range(grid.n_rows) for c in range(grid.n_cols)]
    assert (ownership(grid, every) == 1).all()


def test_kept_cores_never_overlap_and_cover_all_tissue(slide_path: Path) -> None:
    grid = grid_for(slide_path)
    owned = ownership(grid, list(grid.tiles))
    assert owned.max() == 1
    mask = grid.tissue_mask
    ys, xs = np.nonzero(mask.array)
    centres_x = ((xs + 0.5) * mask.downsample[0]).astype(int)
    centres_y = ((ys + 0.5) * mask.downsample[1]).astype(int)
    assert len(xs) > 0 and owned[centres_y, centres_x].all()
    assert 0 < len(grid.tiles) < grid.n_rows * grid.n_cols


def test_min_tissue_fraction_drops_border_tiles(slide_path: Path) -> None:
    touching = grid_for(slide_path)
    mostly = grid_for(slide_path, min_tissue_fraction=0.5)
    assert set(mostly.tiles) < set(touching.tiles)


def test_grid_covers_the_slide_with_stride_tile_minus_overlap(slide_path: Path) -> None:
    grid = grid_for(slide_path, tile_size=256, overlap=32, mpp=0.5)
    stride = 256 - 32
    last = grid.spec(grid.n_rows - 1, grid.n_cols - 1)
    assert (last.xm, last.ym) == ((grid.n_cols - 1) * stride, (grid.n_rows - 1) * stride)
    assert last.xm + 256 >= WIDTH / grid.downsample > last.xm - stride + 256
    assert grid.spec(0, 1).core_model.x0 == stride + 16 == grid.spec(0, 0).core_model.x1


@pytest.mark.parametrize(
    ("mpp", "level", "factor"),
    [(0.25, 0, 1.0), (0.5, 1, 1.0), (1.0, 2, 1.0), (0.6, 1, 1.2), (4.0, 3, 2.0)],
)
def test_read_level_and_resize_factor(
    slide_path: Path, mpp: float, level: int, factor: float
) -> None:
    grid = grid_for(slide_path, mpp=mpp, tile_size=64, overlap=0)
    assert grid.downsample == pytest.approx(mpp / 0.25)
    assert (grid.level, grid.read_factor) == (level, pytest.approx(factor))
    with open_slide(slide_path) as slide:
        tile = read_tile(slide, grid, grid.tiles[0])
    assert tile.shape == (64, 64, 3) and tile.dtype == np.uint8


@pytest.mark.parametrize("mpp", [0.25, 0.5, 1.0])
def test_model_and_level0_coordinates_round_trip(slide_path: Path, mpp: float) -> None:
    grid = grid_for(slide_path, mpp=mpp)
    for spec in grid.tiles:
        assert grid.to_level0(spec.xm, spec.ym) == (spec.x, spec.y)
        assert grid.to_model(spec.x, spec.y) == (spec.xm, spec.ym)
    points = np.random.default_rng(0).uniform(0, 5000, size=(100, 2))
    for x, y in points:
        assert grid.to_model(*grid.to_level0(x, y)) == pytest.approx((x, y))


def test_tiles_read_the_right_region(slide_path: Path) -> None:
    grid = grid_for(slide_path, mpp=0.25, tile_size=128, overlap=0)
    spec = max(grid.tiles, key=lambda s: s.col)
    with open_slide(slide_path) as slide:
        tile = read_tile(slide, grid, spec)
        direct = slide.read_region_rgb((spec.x, spec.y), 0, (128, 128))
    assert np.array_equal(tile, slide.apply_icc(direct) if grid.icc else direct)


def test_missing_mpp_is_an_error_unless_overridden(slide_without_mpp: Path) -> None:
    with pytest.raises(SlideResolutionError, match="mpp_override"):
        grid_for(slide_without_mpp)
    grid = grid_for(slide_without_mpp, mpp_override=0.25)
    assert grid.slide_mpp == 0.25 and grid.tiles


def test_slide_id_is_stable_across_copies(slide_path: Path, tmp_path: Path) -> None:
    copy = tmp_path / "renamed.tiff"
    copy.write_bytes(slide_path.read_bytes())
    assert slide_id(copy) == slide_id(slide_path) == grid_for(slide_path).slide_id
    copy.write_bytes(slide_path.read_bytes() + b"\0")
    assert slide_id(copy) != slide_id(slide_path)


def test_invalid_geometry_is_rejected(slide_path: Path) -> None:
    with pytest.raises(ValueError, match="overlap"):
        grid_for(slide_path, tile_size=64, overlap=64)
