import json
from importlib.resources import files

from pathoinfer.registry.model_card import card_json_schema

REGENERATE = (
    'regenerate: uv run python -c "import json; from pathoinfer.registry.model_card import '
    'card_json_schema; print(json.dumps(card_json_schema(), indent=2))" '
    "> src/pathoinfer/registry/schema/model_card.v1.json"
)


def test_published_schema_matches_model_card() -> None:
    published = json.loads(
        files("pathoinfer.registry").joinpath("schema/model_card.v1.json").read_text()
    )
    assert published == card_json_schema(), REGENERATE
