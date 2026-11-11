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
| `simplify_px` | `1.5` | Douglas-Peucker tolerance in output pixels (topology kept); turns pixel staircases into diagonal runs |
| `smooth` | `2` | Rounds of Chaikin corner cutting after simplifying: each replaces every vertex with points 1/4 and 3/4 along its edges, doubling the vertices, so outlines are curves rather than pixel steps. `0` keeps outlines on pixel edges |
| `min_area_um2` | `0.0` | Remove small objects: drop regions smaller than this, checked on the merged region (before simplifying, so specks cost nothing more) and again on the final polygon |
| `connectivity` | `8` | 8: parts touching only at a pixel corner are one region (a MultiPolygon meeting at the point); 4: separate regions, so diagonal strands break up |
| `fill_holes_um2` | `0.0` | Fill holes smaller than this. A hole that holds another class's region is kept, so classes never overlap; same-class islands inside a filled hole are dropped. `.inf` in YAML fills every hole |
| `band_height` | `2 * tile` | Rows per band, in output pixels |
| `keep_labels` | `False` | Also return the slide's label map as `rasters["labels"]` (-1 where no tile reached; 255 in the TIFF) |

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
   `background_id`, each connected region (`connectivity`, default 8) becomes the exact
   union of its pixels, holes included (`post.vectorise.regions`), so areas equal pixel
   counts.
4. **Merge seams.** Polygons that reach a band boundary are kept aside and, once every
   band is done, unioned with the same-class polygons they touch (STRtree query, then
   `unary_union` of each touching group).
5. **Fill holes, remove small objects.** On whole merged regions (after the seam merge,
   so band boundaries never split a hole or an object), as the two options say.
6. **Clean up.** Drop collinear vertices, simplify, smooth (Chaikin), make valid, keep polygonal parts, drop
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

## Instance (nuclei)

`InstanceStitcher(card, grid, threshold=0.5, marker_threshold=0.4, min_size_px=10,
simplify_px=1.0, smooth=2, workers=0)`; the run config's `instance:` section sets them
(there `workers` defaults to 4).

`workers` > 0 runs steps 1 and 3 for each tile in that many `forkserver` processes
(`tile_columns`; the stitcher thread keeps the softmax and the joining). At most
`2 * workers` tiles are in flight, so the stitch queue still back-pressures the reader,
and results are joined in submission order, so the output equals the serial one. On the
PanNuke mosaic with an RTX 5090: 64 tiles/s on one thread, 160 with 4 processes, 228 with
8 (each slide pays ~0.9 s to start the pool), against 279 tiles/s for the FP16 TensorRT
plan. A worker's error is raised from the next `add` or from `finalize`.

1. **Per tile.** The `np` softmax gives each pixel's nucleus probability, and HoVer-Net's
   watershed on it and the `hv` maps gives an instance map (`post/instances.py`,
   ported from pathotrain so inference matches its validation). Each instance takes the
   majority non-background type of the `nt` argmax under it. Untyped instances are
   dropped.
2. **Ownership.** As for detections, an instance is kept only by the tile whose core holds
   its centroid. A nucleus up to about `overlap` wide is seen whole by its owner and
   reported once; a neighbour that sees it cut has its shifted centroid outside its core.
   Larger objects can be cut or reported twice (warning below the card's
   `recommended_overlap`).
3. **Polygons.** A tile's kept instances are traced together from its label map
   (`post/vectorise.label_pieces`, then `to_features`), so touching nuclei share one
   border line.

`finalize()` returns `StitchResult(features=[], instances=Instances(...))`: NumPy columns
`geometry`, `class_id`, `score` (the type's mean probability), `area_um2` and
measurements `perimeter_um`, `nucleus_probability`.

## Classification heatmaps

`HeatmapStitcher(card, grid, sigma_tiles=0, thresholds=None, ...)` for a
`ClassificationCard` (one `(batch, n_classes)` output per tile):

1. **Average into cores.** A core's probabilities are the mean of every kept tile that
   overlaps it, weighted by overlap area: its own tile counts `stride²`, each side
   neighbour `stride * overlap/2`, each corner neighbour `(overlap/2)²`. Cores of tiles
   that were not kept have no value (NaN).
2. **Smooth (optional).** `sigma_tiles` > 0 applies a Gaussian over the core grid,
   normalised so cores without a value neither count nor change.
3. **Outputs.**
   - `rasters["prob_<class>"]`: float16, one pixel per core, with `scale` the grid stride
     and `origin` half the overlap in level-0 px (exact for interior cores; edge cores
     reach the slide edge, half an overlap further than their pixel).
   - `detections`: one object per kept core (its box), `labels` the argmax class,
     `scores` its probability, `class_probs` every class's probability; writers emit them
     as QuPath detections with `prob_<class>` measurements, for measurement maps.
   - `features`: with `thresholds={"tumour": 0.5}`, region polygons where the class's
     probability reaches the threshold, traced on the core grid by `post.vectorise`
     (`simplify_px` default 1 core, `smooth` 2, `min_area_um2`, `fill_holes_um2`).

Engine runs for classification are not wired yet: the writers take objects from either
`detections` or `features`, and write one `objectType` per run.

## Shared vectoriser

Segmentation and heatmap regions both use `post/vectorise.py`: `regions` (contours with
holes on pixel edges), `merge` (across band seams), `fill_holes`, `to_features`
(simplify and smooth every border once so neighbours share it, scale to level 0,
measure) and `vectorise` (all of them on one whole
mask). Regions are exact pixel unions; with `connectivity` 8 (default) parts that touch
only at a pixel corner are one region, with 4 separate regions. `post/README.md` has
the details and measurements.

## Testing

`tests/stitch/test_heatmap.py` feeds a checkerboard of 3 x 3-tile blocks: the raster
equals a brute-force average that intersects every core with every tile (also with
missing tiles); one hot tile spreads exactly `stride²`, `stride * overlap/2` and
`(overlap/2)²` over `(stride + overlap)²`; the thresholded region is exactly the tumour
cores (one region with connectivity 8, four with 4); smoothing keeps probabilities summing to 1 and gaps empty.

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
