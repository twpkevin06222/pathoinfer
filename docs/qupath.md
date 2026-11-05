# Viewing outputs in QuPath

pathoinfer's GeoJSON follows QuPath's format, and all coordinates are level-0 pixels of
the slide, which is what QuPath uses. Open the same slide file in QuPath and the objects
land where they belong. `docs/outputs.md` describes every field.

Written for QuPath 0.5. QuPath was not available where this was written, so the menu
paths and the script below are not tested; the GeoJSON itself is checked against the
slide with `scripts/qc_overlay.py`.

## Before you start

- Check alignment outside QuPath first:
  `uv run python scripts/qc_overlay.py SLIDE OUTPUT.geojson.gz` draws the objects on the
  thumbnail and on three full-resolution crops.
- Open the slide in QuPath with the same file pathoinfer read. Formats where a reader
  crops to the scanned area (for example MRXS) can shift objects by that area's offset;
  if the overlay script shows them aligned but QuPath does not, compare the image size
  in QuPath's Image tab with `pathoinfer info slide SLIDE`.
- For large detection outputs, write plain `.geojson` (`outputs.gzip: false`), or
  decompress first (`gunzip -k slide_name-01.geojson.gz`), if your QuPath version does
  not import `.gz` files.

## Import objects

1. Open the slide (File > Open, or drag it onto QuPath).
2. File > Import objects from file, and choose `<slide>.geojson`. Dragging the file onto
   the viewer works too.
3. Objects arrive with their class and colour:
   - segmentation regions are **annotations** (editable, one per region, holes kept);
   - detections are **detections** (light, not editable; suited to millions of objects).
4. Measure > Show annotation measurements (or detection measurements) lists each
   object's `area_um2`, `perimeter_um`, `mean_probability`, `score` and `prob_<class>`.

If a class with the same name already exists in the project, QuPath keeps that class's
colour. Set `class_colors` in the run config, or change the colour in the Annotations
tab.

With `outputs.per_class: true` each class has its own file, so you can import only the
classes you need.

## Measurement maps

Measurement maps colour detections by one of their measurements; this is the way to see
model confidence across a slide.

1. Import a detection GeoJSON (above).
2. Measure > Show measurement maps.
3. Choose a measurement: `score` for detection confidence, or `prob_<class>` when the
   model outputs class probabilities. Adjust the colour range in the dialog.

Tile classification (v0.2) will write one detection per tile with `prob_<class>`
measurements, so its heatmaps use the same steps.

## TIFF overlays

`<slide>.labels.tiff` (segmentation, `outputs.tiff: true`) is a pyramidal TIFF at the
segmentation's output resolution, with resolution tags set to its physical pixel size.
Pixel values are class ids, -1 where no tile reached.

To look at it on its own, add it to the project as an image (File > Open): QuPath reads
the pixel size from the resolution tags, so measurements are in microns. It is coarser
than the slide (one pixel is `output_mpp` microns), so it does not share the slide's
pixel coordinates.

To overlay it on the slide, scale it by its level-0 pixels per pixel. The
ImageDescription of the TIFF holds that factor as JSON (`level0_per_px`). A script for
QuPath's script editor (Automate > Show script editor), not run here:

```groovy
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import qupath.lib.gui.viewer.overlays.ImageServerOverlay
import qupath.lib.images.servers.ImageServerProvider

def path = '/path/to/slide_name-01.labels.tiff'
def level0PerPx = 8.0  // "level0_per_px" in the TIFF's ImageDescription

def server = ImageServerProvider.buildServer(path, BufferedImage)
def viewer = getCurrentViewer()
def transform = AffineTransform.getScaleInstance(level0PerPx, level0PerPx)
def overlay = new ImageServerOverlay(viewer, server, transform)
overlay.setOpacity(0.5)
viewer.getCustomOverlayLayers().add(overlay)
```

The polygons in the GeoJSON carry the same information as the label map and need no
scaling, so prefer them for review; the TIFF is for tools that work on rasters.
