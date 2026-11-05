import json
from importlib.resources import files

from pathoinfer.registry.model_card import ModelCard

REGENERATE = (
    'regenerate: uv run python -c "import json; from pathoinfer.registry.model_card import '
    'ModelCard; print(json.dumps(ModelCard.model_json_schema(), indent=2))" '
    "> src/pathoinfer/registry/schema/model_card.v1.json"
)


def test_published_schema_matches_model_card() -> None:
    published = json.loads(
        files("pathoinfer.registry").joinpath("schema/model_card.v1.json").read_text()
    )
    assert published == ModelCard.model_json_schema(), REGENERATE


def test_example_card_validates() -> None:
    card = ModelCard.model_validate(
        {
            "name": "tumour-seg",
            "version": "1",
            "task": "segmentation",
            "format": "onnx",
            "file": "model.onnx",
            "tile_size": 512,
            "tile_mpp": 0.5,
            "preprocessing": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            "classes": [{"index": 0, "name": "background"}, {"index": 1, "name": "tumor"}],
            "outputs": [
                {
                    "name": "logits",
                    "shape": ["batch", 2, 512, 512],
                    "dtype": "float32",
                    "kind": "logits",
                }
            ],
        }
    )
    assert card.schema_version == 1
    assert card.preprocessing.icc
