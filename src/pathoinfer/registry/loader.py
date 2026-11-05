"""Resolve a model reference to a ModelCard and its local model folder. Lands in v0.1.

Resolution order for a local path:

1. A folder with ``model_card.json`` (canonical).
2. A bare ONNX file whose metadata holds the card under the key ``model_card``.
3. A bare ONNX file with only the legacy ``pathotrain`` key: adapted to a ModelCard
   (see ``docs/model_contract.md``) with a DeprecationWarning.
4. Anything else: an error suggesting ``pathoinfer card create``.

``mlflow://`` (``[mlflow]`` extra) and ``hf://`` (``[hf]`` extra) references are
downloaded first and then resolved as a local folder of the same format.
"""

from pathlib import Path

from pathoinfer.registry.model_card import ModelCard


def load_card(ref: str | Path) -> tuple[ModelCard, Path]:
    """Return the card for ``ref`` and the folder its ``file`` is relative to."""
    raise NotImplementedError("lands in v0.1")
