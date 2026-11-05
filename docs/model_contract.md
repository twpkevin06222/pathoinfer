# Model contract

pathoinfer runs any model that comes with a **ModelCard**. pathoinfer owns the schema:
`src/pathoinfer/registry/schema/model_card.v1.json`, generated from
`pathoinfer.registry.model_card.ModelCard`. Producers such as pathotrain conform to the
schema without importing pathoinfer.

## Model folder

```text
my-model/
  model_card.json   canonical card
  model.onnx        the file named by the card's "file" (ONNX, TorchScript, torch.export, TensorRT plan)
```

An ONNX file also embeds the same card as JSON under the metadata key `model_card`, so a
bare `.onnx` file stays self-describing. The metadata key `pathotrain` is training
provenance only.

## ModelCard v1

| Field | Type | Meaning |
|---|---|---|
| `schema_version` | `1` | Card schema version |
| `name`, `version` | str | Model identity, recorded in every output |
| `task` | `segmentation` \| `detection` \| `heatmap` | What the outputs mean |
| `architecture` | str or null | Free text, e.g. `smp:unet` |
| `format` | `onnx` \| `torchscript` \| `torch_export` \| `tensorrt_plan` | How `file` is executed |
| `file` | str | Model file, relative to the folder |
| `input_name` | str, default `image` | Graph input name |
| `tile_size` | int > 0 | Input side in pixels; input is `(batch, 3, tile_size, tile_size)` |
| `tile_mpp` | float > 0 | Microns per pixel the model was trained at |
| `preprocessing.mean`, `.std` | 3 floats each | Per channel, on `[0, 1]` pixels |
| `preprocessing.icc` | bool, default true | Convert the slide's ICC profile to sRGB before inference |
| `classes[]` | `{index, name, color?}` | Output channel (segmentation) or label (detection); `color` is RGB 0-255, default tab20 |
| `outputs[]` | `{name, shape, dtype, kind}` | `kind`: `logits`, `probabilities`, `boxes_xyxy`, `scores`, `labels`; `shape` names dynamic axes |
| `provenance` | object | Free form; the legacy `pathotrain` metadata is kept here |

Preprocessing is fixed apart from these fields: RGB uint8 tiles, `x / 255`, then
`(x - mean) / std`, fed as float32 NCHW.

## Resolution order (`registry/loader.py`)

1. A folder with `model_card.json`.
2. A bare ONNX file with the `model_card` metadata key.
3. A bare ONNX file with only the legacy `pathotrain` key. It is adapted to a ModelCard as
   in the next section, with a DeprecationWarning.
4. Anything else: an error suggesting `pathoinfer card create`.

`mlflow://` and `hf://` references are downloaded, then resolved as a local folder.

`pathoinfer card create` wraps third-party models, and legacy pathotrain ONNX files, in a
model folder. It prefills fields from the `pathotrain` key when present.

## Legacy `pathotrain` key -> ModelCard

| `pathotrain` key | ModelCard field | Rule |
|---|---|---|
| `task` | `task` | Same values (`segmentation`, `detection`) |
| `model` | `architecture` | Copied |
| `input` | none | Free text; replaced by the fixed preprocessing above |
| `mean`, `std` | `preprocessing.mean`, `.std` | Copied |
| none | `preprocessing.icc` | `true`: pathotrain always trains with ICC on |
| `tile_size` | `tile_size` | Copied |
| `tile_mpp` | `tile_mpp` | Copied. `null` (trained on tile folders) cannot be adapted: run `card create` and give the mpp |
| `num_classes` | none | Must equal `len(classes)` |
| `class_names` | `classes[i].name` | Index `i` is the output channel or label. If the names are the defaults `"0"`, `"1"`, ... and `class_map` is set, names come from `class_map` inverted (label -> name), and segmentation label 0 is `background` |
| `class_map` | none | Annotation name -> label; used for names as above, kept in provenance |
| `outputs` | `outputs[]` | Free text, so the adapter uses the task's known layout (below) |
| none | `name` | ONNX file stem |
| none | `version` | `"0"` |
| none | `format`, `file` | `onnx` and the ONNX file name |
| none | `classes[].color` | Unset, so tab20 defaults apply |
| whole object | `provenance.pathotrain` | Copied unchanged |

Known output layouts:

- **segmentation**: `logits`, shape `[batch, num_classes, tile_size, tile_size]`, float32, kind `logits`.
- **detection**: `boxes` `[batch, 300, 4]` float32 kind `boxes_xyxy` (tile pixels);
  `scores` `[batch, 300]` float32 kind `scores` (sigmoid); `labels` `[batch, 300]` int64
  kind `labels`.

## What pathotrain's export must add to write a valid card

- Write `model_card.json` beside the ONNX file, and embed the same JSON under the
  `model_card` metadata key. Keep `pathotrain` as provenance.
- `schema_version: 1`, `name`, `version`, `format: "onnx"`, `file`.
- `preprocessing.icc` (true unless the training config turned ICC off).
- `classes` as `{index, name}` objects with real names, background included for
  segmentation. Optional `color`.
- `outputs` as structured `{name, shape, dtype, kind}` entries instead of free text.
- Always set `tile_mpp`. Refuse to export a model trained on tile folders without one, or
  take it as an argument.
