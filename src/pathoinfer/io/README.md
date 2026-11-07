# pathoinfer - io

Writes a slide's stitched result to files: GeoJSON for viewers such as QuPath, Parquet
for analysis, and pyramidal TIFF for heatmaps. Every coordinate is a level-0 pixel
coordinate of the slide (x right, y down), not WGS84. `docs/outputs.md` describes each
file's fields and columns.

## Contract

Every writer reads the same two inputs, both from `io/objects.py`:

| | |
|---|---|
| `RunInfo.create(card, grid, model_source, output_mpp=None, colors=None)` | The run's metadata: slide id and name, task, model name, version and source, ModelCard schema, slide and model mpp, output mpp, tile size, overlap, tissue area, classes (without background, colour overrides applied), package versions, UTC time. Recorded in every output. |
| `ObjectTable.from_result(result, run, geometry="box")` | One row per object as NumPy columns: shapely `geometry`, `class_id`, `score` (NaN for segmentation), `area_um2` and the task's `measurements`. `geometry` picks box polygons or centre points for detections. |

Objects stay columns from stitcher to file, so a million detections never become a
million Python objects.

## Usage

```python
from pathoinfer.io.objects import ObjectTable, RunInfo
from pathoinfer.io.writers.geojson import write_geojson
from pathoinfer.io.writers.parquet import write_parquet, write_summary
from pathoinfer.io.writers.tiff import write_tiff

run = RunInfo.create(card, grid, "models:/cells@prod", colors=config.class_colors)
objects = ObjectTable.from_result(result, run, geometry=config.detection.geometry)

write_geojson("out/slide_name-01.geojson.gz", objects, run)
write_parquet("out/slide_name-01.objects.parquet", objects, run)
write_summary("out/slide_name-01.summary.parquet", objects, run)
if "labels" in result.rasters:
    write_tiff("out/slide_name-01.labels.tiff", result.rasters["labels"], run)
```

`card` comes from `registry.loader.load_model`, `grid` from `tiling.grid.build_grid` and
`result` from a stitcher's `finalize()`.

Each writer writes `<file>.partial` and renames it when done, so a file that exists is
complete. Each logs the path it wrote at INFO.

## Modules

| Module | Writes |
|---|---|
| `writers/geojson.py` | `write_geojson(path, objects, run, per_class=False)`: QuPath FeatureCollection (`objectType`, `classification`, `measurements`, `object_id`), run metadata in the top-level `properties`, coordinates rounded to 0.1 px. `.geojson` or `.geojson.gz`; `per_class` writes `<stem>.<class>.geojson[.gz]` per class. |
| `writers/parquet.py` | `write_parquet(path, objects, run, geoparquet=False)`: objects table, WKB geometry, zstd, 100k-row row groups, run metadata under the key-value key `pathoinfer`. `write_summary(path, objects, run)`: one row per class with `count`, `mean_score`, `total_area_um2` and `tissue_area_um2`. |
| `writers/tiff.py` | `write_tiff(path, raster, run)`: 2D raster as a tiled zlib BigTIFF pyramid, each level a top-level image (OpenSlide, fastslide and QuPath open it), resolution tags, label maps 8-bit (255 = no tile), run metadata as JSON in the ImageDescription. |
| `objects.py` | `RunInfo`, `ObjectTable`. |
| `interop.py` | `to_geodataframe(path)`: an objects table as a geopandas `GeoDataFrame`, behind the `[geopandas]` extra. The only module that imports geopandas. |

Analysis over many slides' Parquet files lives in `pathoinfer.analysis`
(`class_density`, `score_histogram`); `docs/outputs.md` has examples.

## GeoParquet

`geoparquet=True` writes through DuckDB's spatial extension, which adds the GeoParquet
`geo` metadata. A `geo` entry without a `crs` means longitude and latitude, so the
geometry is tagged with a PROJJSON engineering CRS for level-0 pixels (`pixel_crs`). The
writer reads the metadata back after every write and raises if it is missing or wrong.
The first use downloads the extension, which needs network access. The default, plain
Parquet with WKB, needs neither.

## Testing

`tests/io/` writes the stitched fixtures from `tests/outputs.py` (a disk and an annulus
from the segmentation stitcher, a grid of detections) and reads each file back:

- GeoJSON re-reads with shapely as the same shapes within the 0.1 px rounding, holes
  kept; per-class files keep the global `object_id`s;
- Parquet reads the same through Polars and DuckDB, and WKB decodes to the same shapes
  exactly;
- GeoParquet metadata names the pixel CRS (skipped without the spatial extension);
- the summary counts every class, including classes without objects;
- the TIFF has the expected pyramid levels, tiling, compression and resolution.

`tests/test_analysis.py` runs the analysis queries over two written slides.
