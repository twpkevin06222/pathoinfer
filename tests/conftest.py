import json
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

pytest_plugins = ["tests.outputs"]


@pytest.fixture
def caplog(caplog: pytest.LogCaptureFixture) -> Iterator[pytest.LogCaptureFixture]:
    """Route loguru records into pytest's caplog, as patchology's tests do."""
    handler_id = logger.add(caplog.handler, format="{message}", level=0)
    yield caplog
    logger.remove(handler_id)


@pytest.fixture
def vqa_folder(tmp_path: Path) -> Callable[..., Path]:
    """A VLM model folder (card + ``config.json``, no weights): ``precision``, the
    ``quantization_config`` the weights carry (``quant``) and card ``input`` overrides."""

    def make(precision: str = "bf16", quant: dict[str, Any] | None = None, **inputs: Any) -> Path:
        folder = tmp_path / f"vqa-{precision}"
        folder.mkdir()
        config = {"architectures": ["Qwen3VLForConditionalGeneration"]}
        if quant is not None:
            config["quantization_config"] = quant
        (folder / "config.json").write_text(json.dumps(config))
        card = {
            "task": "vqa",
            "name": "path-vqa",
            "version": "1",
            "architecture": "Qwen3VLForConditionalGeneration",
            "precision": precision,
            "input": {"tile_size": 448, "mpp": 0.5, **inputs},
            "output": {"grounding": True},
        }
        (folder / "model_card.json").write_text(json.dumps(card))
        return folder

    return make


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if shutil.which("nvidia-smi"):
        return
    skip = pytest.mark.skip(reason="no CUDA GPU (nvidia-smi not found)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
