# pathoinfer - stitch

Turns per-tile model outputs into one slide-level result in level-0 pixel coordinates
(x right, y down): shapely geometries with their class and measurements, plus optional
rasters. One stitcher per task.

## Contract

| | |
|---|---|
| `Stitcher.add(specs, outputs)` | One batch: row `i` of every tensor in `outputs` (a runtime's raw outputs, on any device) belongs to `specs[i]`. Batches arrive in any tile order. |
| `Stitcher.finalize()` | Called once after the last batch; returns a `StitchResult`. |
| `StitchResult.features` | `Feature(geometry, class_id, class_name, measurements)`; `geometry` is a shapely `Polygon` or `MultiPolygon` in level-0 pixels. |
| `StitchResult.rasters` | `Raster(array, scale, origin)`: pixel `(col, row)` covers level-0 `origin + scale * (col, row)` to `origin + scale * (col + 1, row + 1)`. |

## Usage

Stitching runs on its own CPU thread, so the model never waits on it. `StitchThread`
feeds it through a bounded queue: `put` blocks while the queue is full, which slows the
reader instead of filling memory. A stitcher error is raised from the next `put` or from
`finalize`.

```python
from pathoinfer.stitch.base import StitchThread
from pathoinfer.stitch.segmentation import SegmentationStitcher

stitcher = SegmentationStitcher(card, grid, output_mpp=2.0, min_area_um2=50)
with StitchThread(stitcher, maxsize=8) as thread:
    for tiles, specs in loader:  # pathoinfer.tiling.dataset.make_loader
        thread.put(specs, runtime.predict(tiles))
    result = thread.finalize()

for feature in result.features:
    print(feature.class_name, feature.measurements["area_um2"], feature.geometry.bounds)
```

`card` is a `SegmentationCard` (from `registry.loader.load_model`), `grid` a `SlideGrid`
(from `tiling.grid.build_grid`) and `runtime` a `Runtime` (from
`runtime.factory.make_runtime`).

## Segmentation

`SegmentationStitcher(card, grid, ...)`:

| Option | Default | Meaning |
|---|---|---|
| `output_mpp` | `2.0` | Resolution the probabilities are blended and polygonised at; never finer than the model's mpp |
| `simplify_px` | `1.5` | Douglas-Peucker tolerance in output pixels (topology kept) |
| `min_area_um2` | `0.0` | Drop polygons smaller than this |
| `band_height` | `2 * tile` | Rows per band, in output pixels |
| `keep_labels` | `False` | Also return the slide's label map as `rasters["labels"]` (-1 where no tile reached) |

How it works (the module docstring has the maths):

1. **Blend.** Each tile's class probabilities (softmax of the logits, unless the card's
   output activation is `probs`) are area-resampled to `output_mpp` and weighted by a
   centre-heavy Gaussian (nnU-Net style, sigma = tile / 8). Weighted probabilities and
   weights are summed into separate canvases and divided at the end, so tile edges, where
   models are least reliable, count least.
2. **Stream in bands.** The canvas is held as horizontal bands. Each band counts the grid
   tiles that overlap it and is finalised when the last one arrives, so memory stays at a
   few bands, not the slide, whatever order the DataLoader workers deliver tiles in.
   Peak canvas memory is logged (`peak_bytes`).
3. **Polygonise each band.** Argmax gives a label map. For every class except the card's
   `background_id`, `cv2.findContours(RETR_CCOMP, CHAIN_APPROX_SIMPLE)` traces each
   8-connected region and its holes through pixel centres; each contour segment is
   widened by half a pixel, so polygons follow pixel edges and areas match pixel counts.
4. **Merge seams.** Polygons that reach a band boundary are kept aside and, once every
   band is done, unioned with the same-class polygons they touch (STRtree query, then
   `unary_union` of each touching group).
5. **Clean up.** Drop collinear vertices, simplify, make valid, keep polygonal parts, drop
   regions under `min_area_um2`, scale to level 0 and round to 0.1 px.

Measurements per feature: `area_um2`, `perimeter_um` and `mean_probability` (the blended
probability of the feature's class over its pixels).

Two pitfalls the code avoids, both covered by tests:

- Never buffer a whole pixel-centre contour as one line. Such a contour can run back over
  itself, and GEOS then drops parts of it; buffering segment by segment does not.
- Drop collinear vertices before simplifying. Otherwise Douglas-Peucker chooses between
  near-tied points and moves corners depending on how a polygon was assembled.

## Detection

`DetectionStitcher(card, grid, score_threshold=None, nms_iou=None)`; the run config's
`detection:` section sets both and the output `geometry`.

1. **Decode.** Boxes are read as the card says (`xyxy` or `cxcywh`, `pixel` or
   `normalised`), converted to `xyxy` and moved to level 0:
   `(tile_origin_model + box) * downsample`. Scores below the threshold (the config's,
   else the card's) are dropped.
2. **Ownership.** A detection is kept only by the tile whose core contains its box
   centre. Cores partition the slide, so an object seen whole by every overlapping tile is
   reported exactly once. That holds for objects whose half-size is at most
   `overlap // 2` model pixels (up to about `overlap` wide): the owning tile's core is
   `overlap // 2` inside the tile, so the owner sees the whole object. A warning is
   logged when the grid's overlap is below the card's `recommended_overlap`.
3. **NMS (optional).** Wider objects are cut by tile edges, shift their centres and can
   be reported twice. With `nms_iou` set, detections centred within `overlap` model px
   of an interior core edge go through class-aware NMS against other tiles' detections
   (STRtree pairs, greedy by score); a tile's own detections are never merged. Two tiles
   see such an object cut at their edges, so the IoU is about `overlap / width`: NMS
   removes the duplicate only when `nms_iou` is below that. More overlap is the
   dependable fix.

`finalize()` returns `StitchResult(features=[], detections=Detections(...))`: NumPy
columns `boxes` (level-0 `xyxy`), `scores`, `labels` and, when the card names a
`class_probs` output, `class_probs`. Writers read them as columns too
(`io.objects.ObjectTable.from_result`), so no Python object exists per detection.
Output formats are in `docs/outputs.md`.

## Other tasks

| Module | Status |
|---|---|
| `heatmap.py` | Stub until pathotrain exports a tile classifier. |

## Testing

`tests/stitch/test_detection.py` uses an ideal detector that reports every object a tile
overlaps, clipped at the tile edge: objects on four-tile corners and edges are reported
once, a dense lattice gives the exact count and boxes, NMS removes duplicates only below
`overlap / width`, and 1 million detections stitch in about a second (measured, one run).

`tests/stitch/test_segmentation.py` builds a synthetic `SlideGrid` and feeds analytic
logits, so expected results are exact:

- a disk centred on the corner of four tiles is one polygon within 2% of its analytic
  area and perimeter (measured: 0.11% and 0.06%);
- an annulus keeps exactly one interior ring;
- banded output equals a single canvas to 1e-4 of area without simplification, and
  within the simplification tolerance with it;
- shuffled tile order gives the same result;
- peak band memory stays under half a full canvas;
- `StitchThread` matches a direct call and re-raises stitcher errors.
