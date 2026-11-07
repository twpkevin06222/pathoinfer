# Outputs

pathoinfer writes four kinds of file per slide. All coordinates are **level-0 pixel
coordinates of the slide: x to the right, y down, origin at the top-left corner.** This is
the convention of pathology viewers such as QuPath. It is not RFC 7946 GeoJSON's WGS84
longitude and latitude, so do not load these files into GIS tools as geographic data.

| File | Writer | Use |
|---|---|---|
| `<slide>.geojson[.gz]` | `io.writers.geojson.write_geojson` | Viewing in QuPath and other viewers |
| `<slide>.objects.parquet` | `io.writers.parquet.write_parquet` | Analysis, one row per object |
| `<slide>.summary.parquet` or `.json` | `io.writers.parquet.write_summary` | Analysis, one row per class (`summary_format`) |
| `<slide>.<name>.tiff` | `io.writers.tiff.write_tiff` | Heatmaps and label maps over the slide. The segmentation label map has the same clean-up as the polygons (holes filled, small regions removed) |
| `<slide>.overlay.png` | `io.writers.overlay.write_overlay` | Segmentation: the whole slide at `overlay_mpp` (default 6) with each class filled at 40% opacity, outlined, and a legend |

Every writer takes a `RunInfo` and, except the TIFF writer, an `ObjectTable` (both in
`io.objects`):

```python
from pathoinfer.io.objects import ObjectTable, RunInfo

run = RunInfo.create(card, grid, model_source="models:/cells@prod", colors=config.class_colors)
objects = ObjectTable.from_result(result, run, geometry=config.detection.geometry)
```

Each writer writes to `<file>.partial` and renames it when done, so a file that exists is
complete.

## Run metadata

`RunInfo` is recorded in every output: the GeoJSON collection's `properties`, the
Parquet key-value metadata key `pathoinfer` (as JSON), the summary's columns and the
TIFF's ImageDescription.

| Field | Meaning |
|---|---|
| `slide_id` | sha256 of the slide's size and first MiB (`tiling.slide_id`) |
| `slide_name` | Slide file name |
| `task` | `segmentation`, `detection` or `classification` |
| `model_name`, `model_version`, `model_source` | From the ModelCard; the source as given (path, `models:/`, `hf://`) |
| `model_card_schema` | ModelCard `schema_version` |
| `slide_mpp`, `model_mpp` | Slide level-0 and model resolution, microns per pixel |
| `output_mpp` | Resolution segmentation was polygonised at; null for other tasks |
| `tile_size`, `overlap` | Model tile size and grid overlap, model pixels |
| `tissue_area_um2` | Tissue area from the grid's tissue mask, µm² |
| `classes` | `{id, name, color}` for every class an object can have (no background) |
| `pathoinfer_version`, `patchology_version` | Installed package versions |
| `created_at` | UTC, ISO 8601 |

## GeoJSON

A FeatureCollection in QuPath's format. QuPath imports it with File > Import objects.

```json
{"type": "FeatureCollection",
 "properties": {"slide_id": "...", "task": "detection", "...": "..."},
 "features": [
  {"type": "Feature",
   "geometry": {"type": "Polygon", "coordinates": [[[57338.6, 24280.8], "..."]]},
   "properties": {
     "objectType": "detection",
     "isLocked": false,
     "classification": {"name": "lymphocyte", "color": [0, 0, 255]},
     "measurements": {"score": 0.91, "area_um2": 9.0, "prob_tumour": 0.07, "prob_lymphocyte": 0.93},
     "object_id": 0}}]}
```

- `objectType` is `annotation` for segmentation regions and `detection` for detections.
- Geometry is a Polygon or MultiPolygon (segmentation, detection boxes) or a Point
  (detections with `geometry: point`). Coordinates are rounded to 0.1 px.
- `measurements` holds `area_um2` and the task's measurements (below). A measurement
  without a value is left out, which QuPath reads as NaN.
- `object_id` is the object's row in the objects Parquet table, so the two files join.
- The top-level `properties` is a GeoJSON foreign member; QuPath ignores it.
- `.geojson.gz` is gzip-compressed. `per_class=True` writes one file per class with
  objects, `<stem>.<class>.geojson[.gz]`, keeping `object_id`s.

Features are streamed to disk in chunks of 10,000. For scale, one run on 1 million
detection boxes took 3.4 s for a 429 MB `.geojson`, and 6.3 s for a 62 MB `.geojson.gz`.

## Objects table (Parquet)

One row per object, built by Polars from NumPy columns. zstd compression, column
statistics, row groups of 100,000 rows.

| Column | Type | Meaning |
|---|---|---|
| `slide_id` | string | As in the run metadata |
| `object_id` | int64 | Row number, also in the GeoJSON |
| `task` | string | |
| `class_id`, `class_name` | int64, string | From the ModelCard classes |
| `score` | float32, null | Detection score; null for segmentation |
| `centroid_x`, `centroid_y` | float64 | Geometry centroid, level-0 px |
| `bbox_x0`, `bbox_y0`, `bbox_x1`, `bbox_y1` | float64 | Geometry bounds, level-0 px |
| `area_um2` | float64 | Region area; a detection's box area, also for point output |
| measurements | float64 | Segmentation: `perimeter_um`, `mean_probability` (over the region's predicted pixels, not filled holes). Detection with a `class_probs` output: `prob_<class>` per class |
| `geometry` | binary | WKB, level-0 px. Decode with `shapely.from_wkb` |
| `model_name`, `model_version` | string | |

The same run metadata is stored as JSON under the key-value metadata key `pathoinfer`
(`polars.read_parquet_metadata(path)["pathoinfer"]`).

### GeoParquet

`write_parquet(..., geoparquet=True)` writes the same table through DuckDB's `spatial`
extension (installed on first use, which needs network access). DuckDB adds the
GeoParquet 1.0 `geo` metadata for the `geometry` column. Without a `crs` entry,
GeoParquet readers take the coordinates as longitude and latitude, so the geometry is
tagged with a PROJJSON engineering CRS: level-0 pixels, x east, y south, one unit =
`slide_mpp` microns (`io.writers.parquet.pixel_crs`). Every write reads the metadata back
and fails if the `geo` entry is missing, is not WKB, or names another CRS. Verified with
DuckDB 2.0.0.

The default is plain Parquet with WKB geometry, which every tool reads.

## Summary table (Parquet)

One row per class, classes without objects included.

| Column | Meaning |
|---|---|
| `slide_id`, `class_id`, `class_name` | |
| `count` | Objects of the class |
| `mean_score` | Mean detection score; null for segmentation |
| `total_area_um2` | Sum of `area_um2` (for segmentation, the class's tissue area) |
| `tissue_area_um2` and the other run metadata fields | Repeated on every row |

With `outputs.summary_format: json` the summary is `<slide>.summary.json` instead: the
run metadata once, and `classes` mapping each class name to its numbers.

```json
{
  "slide_id": "f9b748d9...",
  "slide_name": "slide_name-01.svs",
  "task": "segmentation",
  "tissue_area_um2": 33053697.7,
  "...": "the other run metadata fields",
  "classes": {
    "tumour": {"class_id": 1, "count": 1704, "mean_score": null, "total_area_um2": 12949742.9},
    "stroma": {"class_id": 2, "count": 1956, "mean_score": null, "total_area_um2": 12838110.4}
  }
}
```

## Heatmap TIFF

`write_tiff(path, raster, run)` writes a 2D `Raster` (from `StitchResult.rasters`) as a
pyramidal BigTIFF that OpenSlide, fastslide and QuPath open: every level is its own
top-level image, the full raster first, each next one halved until it fits
one 256 px tile. Floats are area-averaged, integers (label maps) take the nearest pixel.
Tiles are 256 px and zlib-compressed.

Raster pixel `(col, row)` covers level-0 `scale * (col, row)` to `scale * (col + 1, row + 1)`.
The resolution tags give each level's pixel size in pixels per centimetre
(`1e4 / (slide_mpp * scale)` at full resolution), so a viewer that scales by physical size
lays the heatmap over the slide (the readers report it as mpp). Label maps are 8-bit, class
ids with 255 where no tile reached; OpenSlide reads 16-bit images as zeros. Float heatmaps
stay float32, which OpenSlide and fastslide cannot read. The ImageDescription is JSON:
`{"level0_per_px", "um_per_px", "run"}`. Rasters must start at the slide origin.

## Analysis

`pathoinfer.analysis` runs DuckDB over globs of these files and returns Polars frames:

```python
from pathoinfer.analysis import class_density, score_histogram

density = class_density("out/*.objects.parquet", "out/*.summary.parquet", min_score=0.5)
hist = score_histogram("out/*.objects.parquet", bins=20)
```

`class_density` gives `count`, `tissue_area_mm2` and `per_mm2` per slide and class.
`min_score` drops detections below it; segmentation objects have no score and always
count.
`score_histogram` gives `bin`, `lower`, `upper` and `count` per slide and class.

Other questions are plain DuckDB SQL over the same files:

```python
import duckdb
import shapely

# Large objects per slide and class.
duckdb.sql("""
    SELECT slide_id, class_name, count(*) AS n, median(area_um2) AS median_um2
    FROM read_parquet('out/*.objects.parquet', union_by_name = true)
    WHERE area_um2 > 50
    GROUP BY ALL ORDER BY ALL
""").pl()

# Tumour area as a fraction of tissue, per slide (segmentation summaries).
duckdb.sql("""
    SELECT slide_id, total_area_um2 / tissue_area_um2 AS tumour_fraction
    FROM read_parquet('out/*.summary.parquet', union_by_name = true)
    WHERE class_name = 'tumour'
""").pl()

# Objects in a region of one slide, then shapes for plotting.
rows = duckdb.sql("""
    SELECT object_id, class_name, geometry
    FROM 'out/slide_name-01.objects.parquet'
    WHERE bbox_x1 >= 10000 AND bbox_x0 <= 12000 AND bbox_y1 >= 5000 AND bbox_y0 <= 7000
""").pl()
shapes = shapely.from_wkb(rows["geometry"].to_numpy())
```

DuckDB uses the Parquet column statistics to skip row groups that the bounding-box
filter rules out.

## geopandas

`pathoinfer.io.interop.to_geodataframe(path)` reads an objects table into a
`GeoDataFrame` (no CRS: the coordinates are pixels). It needs the `[geopandas]` extra and
is the only place pathoinfer touches geopandas.
