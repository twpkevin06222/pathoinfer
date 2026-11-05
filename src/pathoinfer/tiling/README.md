# pathoinfer - tiling

Turns one slide into an overlapping grid of tiles at the model's resolution, and streams
the tiles to the model from DataLoader workers. Slides are read through patchology
only: patchology opens the slide, reports its mpp, picks the pyramid level and detects
tissue. pathoinfer adds the grid, the tile cores, a stable slide id and the loader.

## Usage

```python
from patchology import open_slide

from pathoinfer.tiling.dataset import make_loader
from pathoinfer.tiling.grid import build_grid


def main() -> None:
    with open_slide("slide_name-01.svs") as slide:
        grid = build_grid(slide, tile_size=512, mpp=0.5, overlap=64)
    print(len(grid.tiles), grid.n_rows, grid.n_cols, grid.level, grid.read_factor)

    for tiles, specs in make_loader(grid, batch_size=32, num_workers=4):
        ...  # tiles: (N, 512, 512, 3) uint8 torch tensor; specs: list[TileSpec]


if __name__ == "__main__":  # required: workers re-import the main script
    main()
```

`build_grid` options:

| Argument | Default | Meaning |
|---|---|---|
| `tile_size` | required | Tile side in model pixels (the card's `input.tile_size`) |
| `mpp` | required | Model resolution in microns per pixel (the card's `input.mpp`) |
| `overlap` | `0` | Overlap between neighbours in model pixels; `0 <= overlap < tile_size` |
| `min_tissue_fraction` | `0.0` | Keep a tile when tissue covers at least this fraction; `0` keeps any tile tissue touches |
| `mpp_override` | `None` | Level-0 mpp for slides that do not report one. Without it such slides raise `SlideResolutionError`. Pass it only when you know the value. |
| `tissue`, `artifacts` | patchology defaults | patchology's `TissueConfig` and `ArtifactConfig` (detection method, pen and blur removal) |
| `icc` | `True` | ICC profile to sRGB, for the tissue thumbnail and every tile |
| `tissue_mask` | `None` | A custom mask instead of detection (any patchology `MaskSource`) |

## Coordinates and reading

Level 0 is the slide at full resolution, `slide_mpp` microns per pixel. Model space is
the slide resampled to the model's `mpp`. One model pixel spans

```text
downsample = mpp / slide_mpp      level-0 pixels
level0 = model * downsample       model = level0 / downsample
```

patchology reads each tile from the highest-resolution pyramid level whose downsample
`ds_L` is at most `downsample` (1% tolerance), reads `tile_size * read_factor` pixels
there with `read_factor = downsample / ds_L`, and resizes them to `tile_size`.
`SlideGrid.level` and `SlideGrid.read_factor` record that choice. Example: a 0.25 mpp
slide with level downsamples 1, 2, 4, 8 and a 0.6 mpp model give `downsample = 2.4`,
level 1 and `read_factor = 1.2`.

## Grid and cores

Tiles start every `stride = tile_size - overlap` model pixels from the slide's top-left
corner, with enough rows and columns to cover the whole slide. Edge tiles may run past
it; patchology pads them with the slide's background colour.

Each tile owns its **core**, the part it writes when stitching:

```text
        overlap // 2                 overlap - overlap // 2
            |                                   |
  +---------+-----------------------------------+---------+
  |  shared |          core of this tile        |  shared |
  +---------+-----------------------------------+---------+
  ^ tile origin                                    tile end ^
```

- A side with a grid neighbour (kept or not) gives up half the overlap to it:
  `overlap // 2` on the left and top, the rest on the right and bottom.
- A side on the slide's edge keeps the full tile extent.

Neighbours split their shared overlap at the same model pixel, so the cores of all grid
tiles partition the slide exactly: every level-0 pixel belongs to one core. Kept tiles'
cores never overlap, and with `min_tissue_fraction = 0` every tissue pixel lies in a kept
core. Counting grid neighbours (not only kept ones) is what keeps two diagonal kept tiles
from both claiming the corner they share.

## Data shapes

`TileSpec`, one tile:

| Field | Meaning |
|---|---|
| `row`, `col` | Grid position |
| `x`, `y`, `size` | Level-0 top-left corner and side (`tile_size * downsample`) |
| `xm`, `ym` | Model-space top-left corner (`col * stride`, `row * stride`) |
| `core` | Level-0 core, a half-open `Box(x0, y0, x1, y1)` clipped to the slide |
| `core_model` | Model-space core |

`SlideGrid`, one slide: the slide's path, id, backend, dimensions and mpp; the model's
`mpp`, `tile_size` and `overlap`; `downsample`, `level` and `read_factor`; `n_rows`,
`n_cols`; the kept `tiles`; and the `tissue_mask` they were selected on.
`grid.spec(row, col)` gives any grid tile, kept or not. `grid.to_level0` and
`grid.to_model` convert coordinates.

## Slide id

`slide_id.slide_id(path)` is the first 32 hex digits of SHA-256 over the file size and
the file's first MiB. Copies and renamed files keep their id; any change to the header
or the size changes it. For MIRAX only the `.mrxs` index file is read.

## Loader

`dataset.TileDataset` gives each DataLoader worker every `num_workers`-th tile
(`get_worker_info`). Each worker opens the slide itself on first use; pickling the dataset
drops any open reader, so reader handles never cross a process boundary. Tiles arrive in
worker order, so use each batch's specs, not its position.

`make_loader` defaults: `pin_memory=True` (when CUDA is available),
`persistent_workers=True`, `prefetch_factor=2`, and `forkserver` workers, because forking
a process that already holds slide decoder threads can deadlock them. forkserver workers
re-import the main script, so a script that starts a loader with workers needs an
`if __name__ == "__main__":` guard and must be a file (not stdin). Each worker takes
about 0.65 s to start (measured on one machine), so reuse a loader across slides where
you can.

## Candidates to move into patchology

- `slide_id`.
- A public read plan (level and factor); today it is rebuilt from
  `get_best_level_for_downsample` and `LEVEL_DOWNSAMPLE_TOLERANCE`.
- A `read_tile` that needs no `TilingResult` (ICC and stain normalisation).
- An edge-padded grid with cores.
- A pickle-safe, lazily opened reader for DataLoader workers.

## Testing and benchmarks

- `tests/tiling/conftest.py` writes a synthetic pyramidal TIFF with tifffile; no real
  slides are in the repo.
- `tests/tiling/test_grid.py` checks that grid cores partition the slide pixel by pixel,
  kept cores cover all tissue, the read level and factor for several mpps, coordinate
  round trips, the missing-mpp error and the slide id.
- `tests/tiling/test_dataset.py` checks that 0, 1 and 3 workers load every tile once.
- `scripts/bench_read.py SLIDE` prints tiles/s for 0, 2, 4 and 8 workers.
