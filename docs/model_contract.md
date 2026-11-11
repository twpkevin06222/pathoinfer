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

A card is one of five variants, chosen by `task`. Each variant accepts only its own
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
| `provenance.parity` | object or null | null | For converted models that passed `pathoinfer convert parity`: `reference` (`name@version`), `tile_set_hash`, `metrics` (worst class per metric), `tolerances`, `checked_at` |
| `provenance.source_metadata` | object | `{}` | Producer metadata kept unchanged, e.g. the legacy `pathotrain` key; converters add `quantisation` or `tensorrt` settings |

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

### Instance

Nucleus instances and their types from HoVer-Net style outputs; the watershed runs in
pathoinfer (`post/instances.py`), not in the graph.

| Field | Type | Meaning |
|---|---|---|
| `background_id` | `0` (the default) | Type channel 0 is background (no nucleus type); never written |
| `output.np` | str, default `np` | `(batch, 2, H, W)` logits: background, nucleus |
| `output.hv` | str, default `hv` | `(batch, 2, H, W)`: each pixel's horizontal and vertical offset from its nucleus's centre, in `[-1, 1]` |
| `output.nt` | str, default `nt` | `(batch, n_classes, H, W)` type logits; channel `i` is class id `i` |
| `output.activation` | `logits` | All three are before any activation |
| `output.n_classes` | int | Must equal `len(classes)` |

### Classification

| Field | Type | Meaning |
|---|---|---|
| `background_id` | int or null, required | Class id of tiles that are not a finding (e.g. normal); null if none |
| `output.tensor` | str, default `logits` | `(batch, n_classes)` output |
| `output.activation` | `logits` \| `probs` | Whether softmax is already applied |
| `output.n_classes` | int | Must equal `len(classes)` |

### Vision-language models (`vqa`)

A VLM is a Hugging Face model folder (`config.json`, safetensors shards, processor
files) that vLLM serves, so its card is a separate variant, `VqaCard`, without the tile
fields above (no `classes`, `normalisation` or `output` tensors). It is written by
`pathoinfer convert vlm-merge` and `convert vlm` (`docs/vlm_quantisation.md`), read with
`registry.loader.load_vlm`; `load_model` refuses it.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `task` | `vqa` | required | Selects the variant |
| `framework`, `file` | `hf`, `config.json` | fixed | A Hugging Face model folder |
| `architecture` | str | required | transformers class, e.g. `Qwen3VLForConditionalGeneration` |
| `precision` | `bf16` \| `fp16` \| `fp8` \| `int8` \| `int4` | `bf16` | The language model's weights (`int4` for W4A16); vision stays bf16 |
| `input.tile_size`, `input.mpp` | int, float | required | How slide crops are made for the model |
| `input.max_pixels` | int or null | null | Cap on what the processor resizes crops to |
| `input.icc` | bool | `true` | ICC profile -> sRGB before cropping |
| `output.grounding` | bool | `false` | Answers contain boxes |
| `output.box_format`, `output.box_units` | `xyxy` \| `cxcywh`; `pixel` \| `normalised` \| `per_mille` | `xyxy`, `per_mille` | How boxes in answers are written (Qwen-VL: per mille) |
| `name`, `version`, `schema_version`, `provenance` | | | As for tile models |

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

## What pathotrain's export writes

pathotrain's segmentation and detection exports write a model folder with a valid card
(`model_card.json` beside the ONNX file and embedded under the `model_card` metadata
key; the old `pathotrain` key stays as `provenance.source_metadata.pathotrain`):

- `input.mpp` and `input.icc` come from the training config; models trained on tile
  folders take them as `--mpp` and `--icc`/`--no-icc`, or the export refuses.
- Classes have real names (`task.class_names` or the class map) and colours (tab20 or the
  top-level `class_colors`); segmentation sets `background_id: 0`.
- `recommended_overlap` is an export argument (`--overlap`, default 0.25), not derived
  from the receptive field. `provenance.training_run_id` is the MLflow run id stored in
  the checkpoint.

Still open: pathotrain has no tile classification export; it needs one that writes a
classification card.
