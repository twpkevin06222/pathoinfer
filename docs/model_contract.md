# Model contract

pathoinfer runs any model that comes as a **model folder**: the model file plus a
`model_card.json`. pathoinfer owns the card's schema. Producers such as pathotrain write
cards that conform to it without importing pathoinfer.

- Schema, in code: `pathoinfer.registry.model_card` (pydantic).
- Schema, published: `src/pathoinfer/registry/schema/model_card.v1.json` (JSON Schema
  2020-12), generated from the code. `tests/registry/test_schema.py` fails if the two
  drift apart.
- Checks: `pathoinfer card validate FOLDER` validates the card, confirms the model file
  exists, and for ONNX compares the graph's tensor names and tile size with the card.

## Model folder

```text
tumour-seg/
  model_card.json
  model.onnx          the file named by "file"
```

An ONNX file may also embed the same card as JSON under the metadata key `model_card`, so
a bare `.onnx` file stays self-describing. The metadata key `pathotrain` is training
provenance only.

## Fields

A card is one of three variants, chosen by `task`. Each variant accepts only its own
fields: a detection card has no `background_id`, and every card's `output` must match its
task.

### Common to every task

| Field | Type | Default | Meaning |
|---|---|---|---|
| `schema_version` | `1` | `1` | Card schema version |
| `name` | str | required | Model name, recorded in every output |
| `version` | str | required | Model version, recorded in every output |
| `task` | `segmentation` \| `detection` \| `classification` | required | Selects the variant |
| `framework` | `onnx` \| `torch` \| `tensorrt` \| `safetensors` | required | How `file` is executed |
| `file` | str | required | Model file, relative to the folder. Suffix must match `framework`: `.onnx`; `.pt` `.pt2` `.pth` `.ts`; `.plan` `.engine`; `.safetensors` |
| `precision` | `fp32` \| `fp16` \| `bf16` \| `int8` \| `fp8` | `fp32` | Weight precision |
| `input.name` | str | `image` | Graph input name |
| `input.tile_size` | int > 0 | required | Tile side in pixels; tiles are square |
| `input.mpp` | float > 0 | required | Microns per pixel the tiles are read at |
| `input.channel_order` | `RGB` \| `BGR` | `RGB` | Channel order of the input |
| `input.layout` | `NCHW` \| `NHWC` | `NCHW` | Input tensor layout |
| `input.dtype` | `float32` \| `float16` \| `uint8` | `float32` | Input tensor dtype |
| `input.normalisation.mean` | 3 floats | required | Per channel, after scaling |
| `input.normalisation.std` | 3 floats > 0 | required | Per channel, after scaling |
| `input.normalisation.scale` | float > 0 | `1/255` | Applied first: `(pixel * scale - mean) / std` |
| `input.icc` | bool | `true` | Convert the slide's ICC profile to sRGB before inference |
| `classes[]` | `{id, name, color}` | required, at least one | `id` is the output channel (segmentation, classification) or label (detection). Ids are `0 .. n-1` in order; names are unique; `color` is `[r, g, b]` in 0-255 |
| `recommended_overlap` | float in `[0, 1)` | required | Tile overlap used when the run config leaves `patchology.tile_overlap` null |
| `provenance.training_run_id` | str or null | null | Training run that produced the weights |
| `provenance.calibration_set_hash` | str or null | null | Hash of the calibration tiles, for quantised variants |
| `provenance.parent_model` | str or null | null | `name@version` this model was converted from |
| `provenance.source_metadata` | object | `{}` | Producer metadata kept unchanged, e.g. the legacy `pathotrain` key |

### Segmentation

| Field | Type | Meaning |
|---|---|---|
| `background_id` | int or null, required | Class id never written as an annotation; null if every class is foreground |
| `output.tensor` | str, default `logits` | `(batch, n_classes, H, W)` output |
| `output.activation` | `logits` \| `probs` | Whether softmax is already applied |
| `output.n_classes` | int | Must equal `len(classes)` |

### Detection

| Field | Type | Meaning |
|---|---|---|
| `output.boxes` | str, default `boxes` | `(batch, n, 4)` boxes |
| `output.scores` | str, default `scores` | `(batch, n)` scores in `[0, 1]` |
| `output.labels` | str, default `labels` | `(batch, n)` class ids |
| `output.box_format` | `xyxy` \| `cxcywh` | Box encoding |
| `output.box_units` | `pixel` \| `normalised` | Tile pixels, or fractions of `tile_size` |
| `output.score_threshold` | float in `[0, 1]` | Default threshold; a run config may override it |
| `output.class_probs` | str or null, default null | Optional `(batch, n, n_classes)` output: each detection's probability per class |

Every label is a foreground class, so a detection card has no `background_id`.

### Classification

| Field | Type | Meaning |
|---|---|---|
| `background_id` | int or null, required | Class id of tiles that are not a finding (e.g. normal); null if none |
| `output.tensor` | str, default `logits` | `(batch, n_classes)` output |
| `output.activation` | `logits` \| `probs` | Whether softmax is already applied |
| `output.n_classes` | int | Must equal `len(classes)` |

## Sources and resolution

`pathoinfer.registry.loader.load_model(source)` returns `(card, model_dir)`.

| Source | Needs | Resolves to |
|---|---|---|
| local path | core | the path |
| `models:/NAME@ALIAS`, `models:/NAME/VERSION`, `models:/NAME@VERSION` | `[mlflow]` | MLflow artifact download into `~/.cache/pathoinfer/mlflow/NAME/REF/`, refreshed on every load |
| `hf://ORG/REPO`, `hf://ORG/REPO@REVISION` | `[hf]` | Hugging Face Hub snapshot |

Every source then goes through the same local resolution:

1. A folder with `model_card.json`.
2. An ONNX file, or a folder holding exactly one, with the `model_card` metadata key.
3. An ONNX file with only the legacy `pathotrain` key, adapted as below, with a
   DeprecationWarning.
4. Anything else: an error suggesting `pathoinfer card create`.

Steps 2 and 3 read ONNX metadata, so they need the `[onnx]` extra.

## `pathoinfer card create`

`pathoinfer card create MODEL OUT [--fields FIELDS.yaml] [--name N] [--version V]`
copies `MODEL` into the new folder `OUT` and writes its card. Fields are merged in this
order, later sources winning:

1. The file: `name` (file stem), `file`, `framework` (from the suffix); for ONNX, the input
   name, the tile size (from a static square input) and the output tensor of a
   single-output model.
2. The legacy `pathotrain` key, mapped as below.
3. An embedded `model_card` key.
4. `--fields` (YAML or JSON, any subset of the card).
5. `--name` and `--version`.

The merged card must validate before anything is written. The error lists every missing
or invalid field.

## Legacy `pathotrain` key -> ModelCard

| `pathotrain` key | ModelCard field | Rule |
|---|---|---|
| `task` | `task` | `segmentation` and `detection` map as is |
| `model` | `provenance.source_metadata.pathotrain.model` | Kept with the rest of the key |
| `input` | none | Free text; replaced by the structured `input` fields |
| `mean`, `std` | `input.normalisation.mean`, `.std` | Copied; `scale` is `1/255` |
| `tile_size` | `input.tile_size` | Copied |
| `tile_mpp` | `input.mpp` | Copied. `null` (trained on tile folders) leaves it unset, so the card fails until `card create --fields` supplies it |
| `num_classes` | `output.n_classes`, `len(classes)` | Used when `class_names` is missing |
| `class_names` | `classes[i].name` | Index `i` is the channel or label. Default names `"0"`, `"1"`, ... are replaced from `class_map` (inverted, label -> name); segmentation id 0 without a name is `background` |
| `class_map` | none | Used for names as above; kept in provenance |
| `outputs` | `output` | Free text; the adapter uses pathotrain's fixed layouts below |
| whole key | `provenance.source_metadata.pathotrain` | Copied unchanged |

Fields the adapter fills from pathotrain's fixed conventions:

| ModelCard field | Value |
|---|---|
| `name`, `version` | file stem, `"0"` |
| `framework`, `file`, `precision` | `onnx`, the ONNX file name, `fp32` |
| `input` | `name: image`, `RGB`, `NCHW`, `float32`, `icc: true` |
| `classes[].color` | tab20, in class order |
| `background_id` (segmentation) | `0` |
| `output` (segmentation) | `tensor: logits`, `activation: logits`, `n_classes: len(classes)` |
| `output` (detection) | `boxes`, `scores`, `labels`; `xyxy`, `pixel`, `score_threshold: 0.5` |
| `recommended_overlap` | `0.25` |

## Gaps pathotrain's export must fill

To write a valid card, pathotrain's export must add these to what it records today:

1. **Write the folder.** Write `model_card.json` beside the ONNX file, and embed the same
   JSON under the `model_card` metadata key. Keep `pathotrain` as provenance.
2. **Identity.** `schema_version: 1`, `name`, `version`, `framework: onnx`, `file`,
   `precision`.
3. **Resolution.** Always set `input.mpp`. A model trained on tile folders must take it as
   an export argument, or the export must refuse.
4. **Structured preprocessing.** `input.name`, `channel_order`, `layout`, `dtype`,
   `normalisation.scale` and `icc`, replacing the free-text `input` string. `icc` must
   reflect the training colour config.
5. **Classes.** `{id, name, color}` per output index with real names, plus
   `background_id` for segmentation (and classification when applicable). Today
   unnamed classes are `"0"`, `"1"`, ... and there are no colours.
6. **Structured outputs.** Replace the free-text `outputs` with the task's `output`
   object: `activation` and `n_classes` for segmentation; tensor names, `box_format`,
   `box_units` and a `score_threshold` for detection.
7. **Overlap.** `recommended_overlap`, from the model's receptive field or validation.
8. **Provenance.** `training_run_id` (the MLflow run id) when known.
9. **Classification.** pathotrain has no tile classification export yet; it needs one
   that writes a classification card.
