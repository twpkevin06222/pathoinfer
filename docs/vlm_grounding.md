# pathoinfer - VLM grounding

This document describes how pathoinfer will find objects on a slide from a question in
words, such as "outline the mitotic figures": Qwen3-VL proposes boxes on crops, and
SAM 3 later turns each box into a mask. The first half of the chain is built; the SAM 3
half is a design.

## The chain

```text
slide --patchology grid--> crops (card.input.tile_size px at card.input.mpp)
      --VllmClient--------> Qwen3-VL answers: [{"bbox_2d": [x0, y0, x1, y1], "label": ...}]
      --parse_boxes-------> boxes in level-0 px, with labels                      built
      --core filter-------> one box per object across overlapping crops            later
      --SAM 3-------------> a mask per box, at SAM's own resolution               later
      --post/vectorise----> polygons in level-0 px                                later
      --ObjectTable-------> GeoJSON / Parquet writers                             exists
```

| Step | Module | Status |
|---|---|---|
| Crops from the tissue grid | `tiling.grid.build_grid`, `read_tile` | Exists |
| Ask Qwen3-VL per crop, answer constrained to a JSON schema | `runtime/remote/vllm.py` (`VllmClient.ask_many`, `json_schema`) | Built |
| Serve Qwen3-VL from its model folder | `serving/vllm.py` (`pathoinfer serve vllm`) | Built |
| Offline batch runs (evaluation, captioning) | `runtime/local/vllm_offline.py` | Built |
| Answer boxes to level-0 px | `runtime/vlm.parse_boxes` | Built |
| Deduplicate boxes across overlapping crops | Reuse the detection rule (below) | Later |
| Box to mask | SAM 3 runtime | Later |
| Mask to polygon | `post/vectorise.py` | Exists |
| Write | `io.objects.ObjectTable` and the writers | Exists |

## Coordinates

Each step changes coordinate space once, and each conversion lives in one place:

| Space | Example | Into the next space |
|---|---|---|
| VLM answer | per mille of the image sent (Qwen3-VL: 0-1000) | `parse_boxes`: divide by the unit, multiply by the crop's level-0 extent, add its origin, clip to the crop |
| Level-0 box | `[x0, y0, x1, y1]` slide px | SAM crop: subtract the SAM crop's origin, scale by its pixels per level-0 px |
| SAM mask | pixels of the SAM input | `post/vectorise.py` polygons, then scale back and add the SAM crop's origin |

The card's `output.box_units` and `box_format` say how to read an answer.
`per_mille` (the default) and `normalised` are relative to the image, so they do not
depend on any resize the model's processor applies. `pixel` answers are in pixels of
the image the model saw. `Crop.image_size` must then be that image's size: the size
after the processor's `max_pixels` resize, if one applied, not the crop's size on disk.

## The built half

```python
from patchology import open_slide

from pathoinfer.registry.loader import load_vlm
from pathoinfer.runtime.remote.vllm import VllmClient
from pathoinfer.runtime.vlm import Crop, parse_boxes
from pathoinfer.tiling.grid import build_grid, read_tile

card, _ = load_vlm("models/pvqa-qwen3vl")
schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "bbox_2d": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
            "label": {"type": "string"},
        },
        "required": ["bbox_2d", "label"],
    },
}
question = "Locate every mitotic figure. Answer with bbox_2d and label for each."

client = VllmClient("http://localhost:8000", concurrency=16)
with open_slide("slides/slide_name-01.svs") as slide:
    grid = build_grid(slide, card.input.tile_size, card.input.mpp, 64, icc=card.input.icc)
    crops = [(read_tile(slide, grid, spec), question) for spec in grid.tiles]
answers = client.ask_many(crops, max_tokens=512, json_schema=schema)

size = (card.input.tile_size, card.input.tile_size)
for spec, answer in zip(grid.tiles, answers, strict=True):
    found = parse_boxes(answer, card.output, Crop(spec.x, spec.y, spec.size, spec.size, size))
    # found.boxes: (n, 4) level-0 px; found.labels: n names
```

This example holds every crop in memory before it asks, so it suits a small region of
interest. A whole-slide run will stream crops through the engine's reader processes,
as tile models do.

## Design of the SAM 3 half

These are the decisions the next step is built on. The open questions follow them.

- **One object, one crop.** Crops overlap, so an object near an edge is found in two
  or more of them. Keep a box only in the crop whose core holds the box's centre,
  the rule `stitch/detection.py` already applies. It is exact for objects up to about
  the overlap wide. The overlap must therefore cover the largest object asked about.
- **SAM 3 gets its own crop.** A box is re-read around itself at SAM 3's input size and
  mpp, not at the VLM's. A small object then gets SAM's full resolution, and a box that
  crosses VLM crops is read whole.
- **Prompts.** Each box is a box prompt. The VLM's label can also be passed as SAM 3's
  text concept, which helps when a box holds several touching objects.
- **Confidence.** Qwen3-VL gives no score per box. SAM 3's predicted mask quality is the
  filter: a box whose best mask scores below a threshold is dropped, which also removes
  hallucinated boxes on empty glass.
- **Masks to polygons through `post/vectorise.py` only**, as for segmentation, so areas
  equal pixel counts and the smoothing rules hold. The polygons are then scaled to
  level-0 px.
- **Outputs.** One row per object in `ObjectTable`. The VLM's label maps to the run's
  classes `{id, name, color}`, and an unknown label is a class of its own, not an error.
  `objectType` is `detection`, and the writers are unchanged.
- **Where SAM 3 runs.** SAM 3 is not served by vLLM, so it is a tile model: an ONNX
  export of its image encoder and mask decoder served by Triton (`runtime/remote/triton.py`)
  or run locally. It runs in the main venv and needs no vLLM.

Open questions:

- The card for a promptable segmenter: a new task beside the three tile tasks, with box
  (and text) prompt inputs, or a pipeline card that names a VLM card and a SAM card.
- Cost: one VLM request per crop. At 448 px and 0.5 mpp a slide has thousands of tissue
  crops, so a tile classifier or heatmap should first choose the crops worth asking
  about.
- Label vocabulary: free-text labels from the VLM against a fixed class list, by exact
  match after normalising (as `convert.vlm_parity.normalise`), or by asking the question
  once per class.

## Tests

- `tests/runtime/test_vlm.py`: every unit and format maps to the same level-0 box,
  clipping, Qwen JSON with labels, and the text fallbacks.
- `tests/runtime/test_vllm_client.py`: the client against a mocked server.
- `tests/serving/test_vllm.py`: the server arguments, and in the vLLM venv, that they are
  vLLM's own.
