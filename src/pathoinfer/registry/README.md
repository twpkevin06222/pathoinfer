# pathoinfer - registry

Owns the model contract: the **ModelCard** that describes a model, and the loader that
turns any model source into a validated **model folder**. pathoinfer owns the schema;
producers such as pathotrain write cards that conform to it without importing pathoinfer.

Field-by-field reference, legacy mapping and the gaps pathotrain's export must fill:
[`docs/model_contract.md`](../../../docs/model_contract.md).

## Model folder

```text
tumour-seg/
  model_card.json     the card
  model.onnx          the file the card's "file" names
```

## Modules

| Module | Responsibility |
|---|---|
| `model_card.py` | `ModelCard`: a union of `SegmentationCard`, `DetectionCard` and `ClassificationCard` keyed on `task`, so a detection card cannot hold a background class or a segmentation output. `parse_card`, `card_json_schema`, `class_colors` (tab20 defaults). |
| `schema/model_card.v1.json` | Published JSON Schema, generated from `model_card.py`. A test fails if they drift. |
| `loader.py` | `load_model(source) -> (card, model_dir)`, `validate_folder`, `graph_problems`, `CardError`. |
| `legacy.py` | Adapts the legacy `pathotrain` ONNX metadata key to card fields. |
| `create.py` | `create_folder`: wraps a model file in a model folder (`pathoinfer card create`). |
| `onnx_info.py` | Reads ONNX metadata and tensor signatures through onnxruntime (`[onnx]` extra). |

## Usage

```python
from pathoinfer.registry.loader import load_model

card, model_dir = load_model("models/tumour-seg")  # local folder or ONNX file
card, model_dir = load_model("models:/tumour-seg@champion")  # MLflow, [mlflow] extra
card, model_dir = load_model("hf://org/tumour-seg@main")  # Hugging Face, [hf] extra

print(card.task, card.input.tile_size, card.input.mpp, [c.name for c in card.classes])
model_file = model_dir / card.file
```

Every source resolves to a local path, then one resolution validates it:

1. a folder with `model_card.json`;
2. an ONNX file (or a folder with exactly one) carrying the card under the metadata key
   `model_card`;
3. an ONNX file with only the legacy `pathotrain` key, adapted with a DeprecationWarning;
4. otherwise a `CardError` suggesting `pathoinfer card create`.

Errors list every problem as `field.path: message`.

## CLI

```bash
# Wrap a model file. Fields come from the ONNX graph, a legacy pathotrain key, an
# embedded card, --fields (YAML or JSON, any subset of the card), then the flags.
pathoinfer card create model.onnx models/tumour-seg --fields fields.yaml --version 1

# Validate a folder; for ONNX also checks the graph's input name, tile size and outputs.
pathoinfer card validate models/tumour-seg
```

## Changing the card

Edit `model_card.py`, then regenerate the published schema:

```bash
uv run python -c "import json; from pathoinfer.registry.model_card import card_json_schema; \
print(json.dumps(card_json_schema(), indent=2))" > src/pathoinfer/registry/schema/model_card.v1.json
```

A change that makes existing cards invalid needs `schema_version: 2`.
