import pickle
from collections import Counter
from pathlib import Path

import pytest
import torch

from patchology import open_slide
from pathoinfer.tiling.dataset import TileDataset, make_loader
from pathoinfer.tiling.grid import SlideGrid, build_grid


def small_grid(slide_path: Path) -> SlideGrid:
    with open_slide(slide_path) as slide:
        return build_grid(slide, tile_size=128, mpp=0.5, overlap=16)


@pytest.mark.parametrize("num_workers", [0, 1, 3])
def test_every_tile_is_loaded_exactly_once(slide_path: Path, num_workers: int) -> None:
    grid = small_grid(slide_path)
    loader = make_loader(grid, batch_size=7, num_workers=num_workers)
    seen: Counter[tuple[int, int]] = Counter()
    for tiles, specs in loader:
        assert tiles.dtype == torch.uint8 and tiles.shape[1:] == (128, 128, 3)
        assert len(specs) == tiles.shape[0]
        seen.update((s.row, s.col) for s in specs)
    assert seen == Counter((s.row, s.col) for s in grid.tiles)
    assert max(seen.values()) == 1


def test_reader_handles_never_cross_processes(slide_path: Path) -> None:
    dataset = TileDataset(small_grid(slide_path))
    next(iter(dataset))
    assert dataset._slide is not None
    clone = pickle.loads(pickle.dumps(dataset))
    assert clone._slide is None
    assert next(iter(clone))[1] == dataset.grid.tiles[0]
    dataset.close()
