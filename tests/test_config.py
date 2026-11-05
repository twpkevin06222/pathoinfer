from pathlib import Path

import pytest
from pydantic import ValidationError

from pathoinfer.config import InferenceConfig, load_config

CONFIGS = Path(__file__).parents[1] / "configs"


@pytest.mark.parametrize("task", ["segmentation", "detection", "heatmap"])
def test_example_config_loads(task: str) -> None:
    cfg = load_config(CONFIGS / f"{task}.yaml")
    assert cfg.task == task
    assert cfg.patchology.icc


@pytest.mark.parametrize(
    "patchology",
    [
        {"slide_mpp": 0.25},  # no override: slides without a resolution are refused
        {"tissue": {"methd": "otsu"}},
        {"tissue": {"method": "nope"}},
        {"tile_overlap": 1.0},
    ],
)
def test_bad_patchology_input_is_rejected(patchology: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(
            {"task": "detection", "model": "m.onnx", "patchology": patchology}
        )


def test_class_colors_must_be_rgb_bytes() -> None:
    with pytest.raises(ValidationError):
        InferenceConfig.model_validate(
            {"task": "segmentation", "model": "m.onnx", "class_colors": {"tumor": [256, 0, 0]}}
        )
