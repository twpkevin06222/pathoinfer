# pathoinfer - post

Post-processing shared by the stitchers. Today that is one module, `vectorise.py`: label
masks to polygons. The segmentation stitcher (per band, then across band seams) and the
heatmap stitcher (thresholded core grids) both use it, so their polygons follow the same
rules.

| Module | Status |
|---|---|
| `vectorise.py` | Masks to level-0 polygons with holes; used by `stitch/segmentation.py` and `stitch/heatmap.py` |
| `nms.py` | Empty stub. Cross-tile NMS for detections lives in `stitch/detection.py` (`_nms`) |
| `smoothing.py` | Empty stub. Polygon smoothing is `vectorise.chaikin`; heatmap smoothing is in `stitch/heatmap.py` |

## Usage

```python
import numpy as np
from pathoinfer.post.vectorise import vectorise

mask = np.zeros((100, 100), bool)
mask[20:60, 30:80] = True  # one region, 40 x 50 mask pixels
mask[35:40, 50:55] = False  # with a 5 x 5 hole
probs = np.where(mask, 0.9, 0.1).astype(np.float32)

(feature,) = vectorise(
    mask,
    probs,
    class_id=1,
    class_name="tumour",
    level0_per_px=8.0,  # one mask pixel = 8 x 8 level-0 px
    slide_mpp=0.25,  # microns per level-0 pixel
)
feature.geometry.bounds  # (240.0, 160.0, 640.0, 480.0), level-0 px
feature.measurements  # area_um2 7894.3, perimeter_um 390.9, mean_probability 0.9
```

The pixel count gives 7900 µm² (1975 px of 64 level-0 px² at 0.25 µm). With
`simplify_px=0, smooth=0` the polygon measures exactly that; the default smoothing rounds
each corner by about a pixel and measures 7894.3.

Mask pixel `(col, row)` covers level-0 `origin + level0_per_px * (col, row)` to
`origin + level0_per_px * (col + 1, row + 1)`; pass `origin` when the mask does not start
at the slide's corner (heatmap core grids start half an overlap in).

## Steps

Each step is public, so a stitcher can run them on band pieces and merge in between.

| Function | Does |
|---|---|
| `regions(mask, values, class_id, top, connectivity=8)` | One `Piece` per connected region, with holes, in mask pixels (`top` offsets the rows of a band): the exact union of its pixels. `Piece` keeps the sum of `values` and the pixel count for `mean_probability`. |
| `merge(pieces, connectivity=8)` | Unions same-class pieces that touch (a region cut by band boundaries): along an edge, or with 8 also at a corner. |
| `fill_holes(pieces, max_px)` | Fills holes under `max_px` mask px² unless another class's region lies inside, and drops same-class islands inside the filled holes. |
| `to_feature(piece, name, level0_per_px=..., slide_mpp=..., origin=..., simplify_px=1.5, smooth=2, min_area_um2=0)` | Drops collinear vertices, simplifies (Douglas-Peucker, topology kept), smooths (edges split to at most 4 px, `chaikin`, then simplified at 0.1 px), makes valid, scales to level 0, rounds to 0.1 px, drops regions under `min_area_um2`, and measures `area_um2`, `perimeter_um` and `mean_probability`. |
| `vectorise(mask, values, class_id, name, ...)` | All of the above on one whole mask, plus `fill_holes_um2` and `connectivity`. |
| `chaikin(geometry, rounds)` | Chaikin corner cutting: each round doubles the vertices and cuts a quarter of each edge from every corner. `to_feature` splits edges to at most 4 px first, so a cut is at most about a pixel. |
| `polygonal(geometry)` | The Polygon or MultiPolygon parts of a geometry, or None. |

## How outlines are traced

A region is the exact union of its pixels. `cv2.connectedComponents` labels the regions;
each row's runs of one label become rectangles, stacked over rows where a run repeats,
and each region's rectangles are unioned. Areas equal pixel counts, outlines lie on pixel
edges, and holes keep their corners. On a 2048 x 2048 noisy mask this takes 2.0 s, where
the previous tracing (contours through pixel centres, widened by half a pixel) took
11.4 s and came out 2-4% too large.

`connectivity` decides whether parts that touch only at a pixel corner are one region:

| | 8 (default) | 4 |
|---|---|---|
| Two squares touching at a corner | one region, a MultiPolygon meeting at that point | two regions |
| A one-pixel diagonal line | one region | one region per pixel |
| Background reaching outside only through a corner | outside | a hole |

On a real slide (dark-pixel model at 2 µm, `min_area_um2: 500`) 8 kept 9.44 mm² in 607
regions and 4 kept 8.94 mm² in 851: under 4, diagonal strands of tissue break into
pixels and fall under `min_area_um2`.

Smoothing works at pixel scale. Simplification turns pixel staircases into diagonal runs;
edges are then split to at most 4 px so that Chaikin cuts at most about a pixel from
each corner per round, and a final 0.1 px simplification drops the near-collinear points
it leaves. A 40 x 50 px rectangle keeps 99.9% of its area; the synthetic slide's ellipse
keeps area and perimeter within 0.2% of the analytic shape with 63 vertices. Use
`smooth=0` for shapes that should keep sharp corners.

## Testing

`tests/post/test_vectorise.py` checks that regions equal the mask's pixels exactly (random
masks, both connectivities), that holes keep their corners, that corner contacts and band
seams follow the connectivity, and that smoothing cuts about a pixel whatever the size.
Through its callers: `tests/stitch/test_segmentation.py`
(analytic disk and annulus, band seams, hole filling, small objects, smoothing),
`tests/stitch/test_heatmap.py` (checkerboard regions) and the pinned synthetic-slide
results in `tests/engine/test_regression.py`.
