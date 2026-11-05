import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from pathoinfer.registry.model_card import SegmentationCard, class_colors, parse_card


def test_default_colours_cycle_the_colormap() -> None:
    names = [f"c{i}" for i in range(21)]
    colors = class_colors(names, cmap="tab20")
    assert colors["c0"] == (31, 119, 180)  # tab20's first colour
    assert len({colors[n] for n in names[:20]}) == 20
    assert colors["c20"] == colors["c0"]


def test_overrides_replace_default_colours() -> None:
    colors = class_colors(["background", "tumor"], {"tumor": (255, 0, 0)})
    assert colors["tumor"] == (255, 0, 0)
    assert colors["background"] == class_colors(["background"])["background"]


def test_override_for_unknown_class_is_rejected() -> None:
    with pytest.raises(ValueError, match="tumour"):
        class_colors(["background", "tumor"], {"tumour": (255, 0, 0)})


FIXTURE = Path(__file__).parents[1] / "fixtures" / "tiny_seg" / "model_card.json"


def card(**changes: object) -> dict[str, Any]:
    return json.loads(FIXTURE.read_text()) | changes


def test_fixture_card_is_a_segmentation_card() -> None:
    parsed = parse_card(card())
    assert isinstance(parsed, SegmentationCard)
    assert parsed.background_id == 0
    assert parsed.output.tensor_names() == ("logits",)


DETECTION_OUTPUT = {
    "box_format": "xyxy",
    "box_units": "pixel",
    "score_threshold": 0.5,
}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"task": "detection", "output": DETECTION_OUTPUT}, "background_id"),
        ({"output": DETECTION_OUTPUT}, "activation"),
        ({"classes": [{"id": 1, "name": "tumor", "color": [1, 2, 3]}]}, "class ids"),
        ({"output": {"activation": "logits", "n_classes": 3}}, "3 classes"),
        ({"background_id": 5}, "background_id 5"),
        ({"file": "model.pt"}, "must end with .onnx"),
        (
            {"classes": [{"id": 0, "name": "bg"}, {"id": 1, "name": "t", "color": [0, 0, 0]}]},
            "color",
        ),
    ],
    ids=[
        "detection-has-no-background",
        "output-matches-task",
        "contiguous-ids",
        "n-classes",
        "background-is-a-class",
        "file-suffix",
        "color-required",
    ],
)
def test_invalid_cards_are_rejected(changes: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        parse_card(card(**changes))
