"""``pathoinfer card create``: wrap a model file in a model folder with a valid card.

Fields come from, lowest priority first: the ONNX graph (input name, tile size, the
output tensor of a single-output model), the legacy ``pathotrain`` key, an embedded
``model_card`` key, then the user's fields. The merged card must validate before anything
is written.
"""

import json
import shutil
from pathlib import Path
from typing import Any

from pathoinfer._logging import log_call
from pathoinfer.registry import legacy
from pathoinfer.registry.loader import CARD_FILE, CARD_KEY, LEGACY_KEY, CardError, parse_or_raise
from pathoinfer.registry.model_card import FRAMEWORK_SUFFIXES, ModelCard
from pathoinfer.registry.onnx_info import read_onnx


@log_call
def create_folder(model: Path, out: Path, fields: dict[str, Any]) -> ModelCard:
    """Copy ``model`` into the new folder ``out`` beside a ``model_card.json``.

    Raises:
        CardError: ``out`` is not empty, or the merged fields are not a valid card.
    """
    if out.exists() and any(out.iterdir()):
        raise CardError(f"{out} is not empty")
    card_fields = _deep_merge(_prefill(model), fields)
    card = parse_or_raise(card_fields, f"the card for {model}")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model, out / card.file)
    (out / CARD_FILE).write_text(card.model_dump_json(indent=2) + "\n")
    return card


def _prefill(model: Path) -> dict[str, Any]:
    framework = next((f for f, s in FRAMEWORK_SUFFIXES.items() if model.name.endswith(s)), None)
    fields: dict[str, Any] = {"name": model.stem, "file": model.name}
    if framework is not None:
        fields["framework"] = framework
    if framework != "onnx":
        return fields
    info = read_onnx(model)
    if len(info.inputs) == 1:
        name, shape = next(iter(info.inputs.items()))
        fields["input"] = {"name": name}
        if len(shape) == 4 and isinstance(shape[2], int) and shape[2] == shape[3]:
            fields["input"]["tile_size"] = shape[2]
    if len(info.outputs) == 1:
        fields["output"] = {"tensor": next(iter(info.outputs))}
    if LEGACY_KEY in info.metadata:
        fields = _deep_merge(
            fields, legacy.card_fields(json.loads(info.metadata[LEGACY_KEY]), model)
        )
    if CARD_KEY in info.metadata:
        fields = _deep_merge(fields, json.loads(info.metadata[CARD_KEY]))
    return fields


def _deep_merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
