"""Read a SlideGrid's tiles in DataLoader worker processes. The one module that imports
torch at the top, so ``import pathoinfer`` and the CLI stay light.

Each worker reads every ``num_workers``-th tile (``get_worker_info``) and opens the slide
itself, on first use; reader handles never cross a process boundary (pickling drops them).
Tiles arrive in worker order, not grid order: use each batch's specs, not its position.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from patchology import SlideReader, open_slide
from pathoinfer.tiling.grid import SlideGrid, TileSpec, read_tile

Item = tuple[NDArray[np.uint8], TileSpec]
Batch = tuple[torch.Tensor, list[TileSpec]]


class TileDataset(IterableDataset[Item]):
    def __init__(self, grid: SlideGrid) -> None:
        self.grid = grid
        self._slide: SlideReader | None = None

    def __iter__(self) -> Iterator[Item]:
        info = get_worker_info()
        tiles = self.grid.tiles if info is None else self.grid.tiles[info.id :: info.num_workers]
        slide = self._open()
        for spec in tiles:
            yield read_tile(slide, self.grid, spec), spec

    def close(self) -> None:
        if self._slide is not None:
            self._slide.close()
            self._slide = None

    def _open(self) -> SlideReader:
        if self._slide is None:
            self._slide = open_slide(self.grid.slide_path, self.grid.backend)  # type: ignore[arg-type]
        return self._slide

    def __getstate__(self) -> dict[str, Any]:
        return {**self.__dict__, "_slide": None}


def collate(batch: list[Item]) -> Batch:
    """``(N, H, W, 3)`` uint8 tiles and their specs."""
    return torch.from_numpy(np.stack([tile for tile, _ in batch])), [spec for _, spec in batch]


def make_loader(
    grid: SlideGrid,
    batch_size: int = 32,
    num_workers: int = 4,
    *,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    multiprocessing_context: str = "forkserver",
) -> DataLoader[Item]:
    """A DataLoader over ``grid``'s tiles. Workers start from a fresh interpreter
    (``forkserver``): forking a process that already holds slide decoder threads can
    deadlock them. With ``num_workers=0`` tiles are read in the calling process."""
    workers: dict[str, Any] = {}
    if num_workers > 0:
        workers = {
            "prefetch_factor": prefetch_factor,
            "persistent_workers": persistent_workers,
            "multiprocessing_context": multiprocessing_context,
        }
    return DataLoader(
        TileDataset(grid),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=pin_memory and torch.cuda.is_available(),
        **workers,
    )
