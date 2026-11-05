"""Resolve a model source to a validated ModelCard and its model folder.

Sources:

- a local path (core),
- ``models:/NAME@ALIAS`` or ``models:/NAME/VERSION`` (``NAME@3`` also works), downloaded
  from MLflow with the ``[mlflow]`` extra,
- ``hf://ORG/REPO`` or ``hf://ORG/REPO@REVISION``, downloaded from the Hugging Face Hub
  with the ``[hf]`` extra.

Remote sources download to a local folder first, so every source ends up in
:func:`resolve_local`, which tries in order:

1. a folder with ``model_card.json`` (canonical);
2. an ONNX file (or a folder holding exactly one) whose metadata has the card under
   ``model_card``;
3. an ONNX file with only the legacy ``pathotrain`` key: adapted with a DeprecationWarning;
4. anything else: a CardError suggesting ``pathoinfer card create``.
"""

import json
import shutil
import warnings
from pathlib import Path
from typing import Any, get_args

from pydantic import ValidationError

from pathoinfer._extras import extra, require
from pathoinfer.registry import legacy
from pathoinfer.registry.model_card import ModelCard, Task, parse_card
from pathoinfer.registry.onnx_info import read_onnx

CARD_FILE = "model_card.json"
CARD_KEY = "model_card"
LEGACY_KEY = "pathotrain"
CACHE_DIR = Path.home() / ".cache" / "pathoinfer"


class CardError(ValueError):
    """A model source has no valid ModelCard."""


def load_model(source: str | Path, cache_dir: Path = CACHE_DIR) -> tuple[ModelCard, Path]:
    """Return ``(card, model_dir)``; the model file is ``model_dir / card.file``."""
    text = str(source)
    if text.startswith("models:/"):
        return resolve_local(_download_mlflow(text, cache_dir))
    if text.startswith("hf://"):
        return resolve_local(_download_hf(text))
    return resolve_local(Path(source))


def resolve_local(path: Path) -> tuple[ModelCard, Path]:
    if path.is_dir() and (path / CARD_FILE).exists():
        return validate_folder(path), path
    onnx = _single_onnx(path)
    meta = read_onnx(onnx).metadata
    if CARD_KEY in meta:
        return _checked(json.loads(meta[CARD_KEY]), onnx.parent, f"{onnx}:{CARD_KEY}"), onnx.parent
    if LEGACY_KEY in meta:
        warnings.warn(
            f"{onnx} has only the legacy '{LEGACY_KEY}' metadata; wrap it once with "
            "`pathoinfer card create` to get a model folder",
            DeprecationWarning,
            stacklevel=2,
        )
        fields = legacy.card_fields(json.loads(meta[LEGACY_KEY]), onnx)
        return _checked(fields, onnx.parent, f"{onnx}:{LEGACY_KEY}"), onnx.parent
    raise CardError(f"{onnx} has no model card; create one with `pathoinfer card create`")


def validate_folder(folder: Path) -> ModelCard:
    """Parse ``folder/model_card.json`` and check the model file it names exists."""
    try:
        data = json.loads((folder / CARD_FILE).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise CardError(f"cannot read {folder / CARD_FILE}: {e}") from e
    return _checked(data, folder, str(folder / CARD_FILE))


def graph_problems(card: ModelCard, folder: Path) -> list[str]:
    """How an ONNX model's graph disagrees with its card: tensor names and tile size."""
    if card.framework != "onnx":
        return []
    info = read_onnx(folder / card.file)
    shape = info.inputs.get(card.input.name)
    if shape is None:
        return [f"input {card.input.name!r} is not a graph input {list(info.inputs)}"]
    problems = []
    side = shape[2:4] if card.input.layout == "NCHW" else shape[1:3]
    if any(isinstance(s, int) and s != card.input.tile_size for s in side):
        problems.append(f"graph input {shape} does not take {card.input.tile_size} px tiles")
    missing = [n for n in card.output.tensor_names() if n not in info.outputs]
    if missing:
        problems.append(f"outputs {missing} are not graph outputs {list(info.outputs)}")
    return problems


def parse_or_raise(data: Any, origin: str) -> ModelCard:
    """:func:`parse_card`, with errors as a CardError listing ``field.path: problem``."""
    task = data.get("task") if isinstance(data, dict) else None
    if task not in get_args(Task):
        raise CardError(f"{origin}: task must be one of {', '.join(get_args(Task))}, got {task!r}")
    try:
        return parse_card(data)
    except ValidationError as e:
        problems = [
            f"  {'.'.join(str(part) for part in err['loc'][1:]) or '(card)'}: {err['msg']}"
            for err in e.errors()
        ]
        raise CardError(f"{origin} is not a valid {task} card:\n" + "\n".join(problems)) from e


def _checked(data: Any, folder: Path, origin: str) -> ModelCard:
    card = parse_or_raise(data, origin)
    if not (folder / card.file).is_file():
        raise CardError(f"{origin} names {card.file!r}, which is not in {folder}")
    return card


def _single_onnx(path: Path) -> Path:
    if path.is_file() and path.suffix == ".onnx":
        return path
    if path.is_dir():
        found = sorted(path.glob("*.onnx"))
        if len(found) == 1:
            return found[0]
    if not path.exists():
        raise CardError(f"{path} does not exist")
    raise CardError(
        f"{path} is not a model folder (no {CARD_FILE}) or a single ONNX file; "
        "create one with `pathoinfer card create`"
    )


def _download_mlflow(uri: str, cache_dir: Path) -> Path:
    artifacts = require("mlflow.artifacts", extra("mlflow"))
    name, _, ref = uri.removeprefix("models:/").partition("@")
    if ref.isdigit():  # mlflow spells versions models:/NAME/VERSION
        uri, ref = f"models:/{name}/{ref}", f"v{ref}"
    dst = cache_dir / "mlflow" / name.replace("/", "-") / (ref or "latest")
    shutil.rmtree(dst, ignore_errors=True)  # an alias may now point at different files
    dst.mkdir(parents=True)
    return Path(artifacts.download_artifacts(artifact_uri=uri, dst_path=str(dst)))


def _download_hf(uri: str) -> Path:
    hub = require("huggingface_hub", extra("hf"))
    repo, _, revision = uri.removeprefix("hf://").partition("@")
    return Path(hub.snapshot_download(repo_id=repo, revision=revision or None))
