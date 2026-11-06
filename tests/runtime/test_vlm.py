import base64
from pathlib import Path

import cv2
import numpy as np
import pytest

from pathoinfer.registry.model_card import VqaOutput
from pathoinfer.runtime.vlm import Crop, chat_messages, image_url, parse_boxes

# A 1024 px image of a 2048 px level-0 region at (1000, 2000): one image px = 2 level-0 px.
CROP = Crop(x=1000, y=2000, width=2048, height=2048, image_size=(1024, 1024))
PER_MILLE = VqaOutput(grounding=True)


def test_qwen_json_boxes_map_through_the_crop_to_level0() -> None:
    answer = """```json
[{"bbox_2d": [0, 0, 500, 250], "label": "tumour"},
 {"bbox_2d": [500, 500, 1000, 1000], "label": "lymphocyte"}]
```"""
    grounding = parse_boxes(answer, PER_MILLE, CROP)
    assert grounding.labels == ["tumour", "lymphocyte"]
    np.testing.assert_allclose(
        grounding.boxes, [[1000, 2000, 2024, 2512], [2024, 3024, 3048, 4048]]
    )


@pytest.mark.parametrize(
    ("output", "box"),
    [
        (VqaOutput(grounding=True, box_units="pixel"), [512, 512, 1024, 1024]),
        (VqaOutput(grounding=True, box_units="normalised"), [0.5, 0.5, 1.0, 1.0]),
        (VqaOutput(grounding=True, box_format="cxcywh"), [750, 750, 500, 500]),
    ],
)
def test_every_unit_and_format_gives_the_same_level0_box(
    output: VqaOutput, box: list[float]
) -> None:
    grounding = parse_boxes(str(box), output, CROP)
    np.testing.assert_allclose(grounding.boxes, [[2024, 3024, 3048, 4048]])
    assert grounding.labels == [""]


def test_boxes_are_clipped_to_the_crop_and_corners_ordered() -> None:
    grounding = parse_boxes('{"bbox": [1100, -50, 900, 300]}', PER_MILLE, CROP)
    np.testing.assert_allclose(
        grounding.boxes, [[1000 + 0.9 * 2048, 2000, 3048, 2000 + 0.3 * 2048]]
    )


def test_text_answers_fall_back_to_bracketed_then_bare_numbers() -> None:
    two = parse_boxes("tumour at [100, 200, 300, 400] and [500,600,700,800].", PER_MILLE, CROP)
    assert len(two) == 2
    one = parse_boxes("box 100 200 300 400", PER_MILLE, CROP)
    np.testing.assert_allclose(
        one.boxes, parse_boxes("[100, 200, 300, 400]", PER_MILLE, CROP).boxes
    )
    none = parse_boxes("no tumour here", PER_MILLE, CROP)
    assert len(none) == 0 and none.boxes.shape == (0, 4)


def test_images_go_as_data_urls_or_local_file_uris(tmp_path: Path) -> None:
    rgb = np.zeros((8, 6, 3), np.uint8)
    rgb[..., 0] = 255  # red
    url = image_url(rgb)
    assert url.startswith("data:image/png;base64,")
    png = np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8)
    np.testing.assert_array_equal(
        cv2.cvtColor(cv2.imdecode(png, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), rgb
    )

    path = tmp_path / "crop.jpg"
    cv2.imwrite(str(path), rgb)
    assert image_url(path).startswith("data:image/jpeg;base64,")
    assert image_url(path, local=True) == path.resolve().as_uri()


def test_chat_message_is_image_then_question_after_the_system_prompt() -> None:
    assert chat_messages("data:x", "tumour?") == [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:x"}},
                {"type": "text", "text": "tumour?"},
            ],
        }
    ]
    assert chat_messages("data:x", "tumour?", system="Be brief.")[0] == {
        "role": "system",
        "content": "Be brief.",
    }
