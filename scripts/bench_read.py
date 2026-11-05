"""Tile read throughput of pathoinfer.tiling.dataset for several worker counts::

    uv run python scripts/bench_read.py SLIDE [--workers 0 2 4 8] [--max-tiles 2048]

Reads the first ``--max-tiles`` tissue tiles in grid order. One untimed pass first warms
the OS page cache, so the numbers are decode + resize + transfer, not disk; pass ``--cold``
to skip it. Throughput excludes worker start-up (reported separately). Each run checks it
read every tile once at the right shape and sums the pixels, so lazily skipped work shows.
"""

import argparse
import os
import time
from dataclasses import replace
from pathlib import Path

from patchology import open_slide
from pathoinfer.tiling.dataset import make_loader
from pathoinfer.tiling.grid import SlideGrid, build_grid


def run(grid: SlideGrid, workers: int, batch_size: int) -> tuple[float, float, int, int]:
    """Start-up seconds, tiles/s after the first batch, tiles read, pixel checksum."""
    loader = make_loader(grid, batch_size=batch_size, num_workers=workers)
    start = time.perf_counter()
    first = None
    count = after_first = checksum = 0
    seen = set()
    for tiles, specs in loader:
        if first is None:
            first = time.perf_counter()
        else:
            after_first += len(specs)
        assert tiles.shape[1:] == (grid.tile_size, grid.tile_size, 3), tiles.shape
        checksum += int(tiles.sum())
        count += len(specs)
        seen.update((s.row, s.col) for s in specs)
    end = time.perf_counter()
    assert first is not None and count == len(grid.tiles) == len(seen), (count, len(seen))
    return first - start, after_first / (end - first), count, checksum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("slide", type=Path)
    parser.add_argument("--mpp", type=float, default=0.5)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tiles", type=int, default=2048)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    parser.add_argument("--cold", action="store_true", help="skip the page-cache warm-up")
    args = parser.parse_args()

    started = time.perf_counter()
    with open_slide(args.slide) as slide:
        grid = build_grid(slide, args.tile_size, args.mpp, args.overlap)
    print(
        f"grid: {len(grid.tiles)} tissue tiles of {grid.n_rows * grid.n_cols}, "
        f"{args.tile_size} px at {args.mpp} mpp (slide {grid.slide_mpp:g} mpp, "
        f"level {grid.level}, read factor {grid.read_factor:.3g}), "
        f"built in {time.perf_counter() - started:.1f} s"
    )
    grid = replace(grid, tiles=grid.tiles[: args.max_tiles])
    print(f"cpus {os.cpu_count()}, load {os.getloadavg()[0]:.1f}, timing {len(grid.tiles)} tiles")
    if not args.cold:
        run(grid, 0, args.batch_size)

    checksums = set()
    print(f"{'workers':>7}  {'start-up s':>10}  {'tiles/s':>8}")
    for workers in args.workers:
        startup, rate, _, checksum = run(grid, workers, args.batch_size)
        checksums.add(checksum)
        print(f"{workers:>7}  {startup:>10.2f}  {rate:>8.0f}")
    assert len(checksums) == 1, f"worker counts read different pixels: {checksums}"
    mean = next(iter(checksums)) / (len(grid.tiles) * args.tile_size**2 * 3)
    print(f"every run read the same pixels (mean value {mean:.1f})")


if __name__ == "__main__":
    main()
