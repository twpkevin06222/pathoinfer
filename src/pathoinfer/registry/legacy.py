"""Adapt the legacy ``pathotrain`` ONNX metadata key to ModelCard fields.

pathotrain exports before the ModelCard store their metadata under that key. This maps
what they carry and fills the rest with pathotrain's fixed conventions; see the mapping
table in ``docs/model_contract.md``. The result is a partial card: ``tile_mpp: null``
(models trained on tile folders) leaves ``input.mpp`` unset, and validation names it.
"""

from pathlib import Path
from typing import Any

from pathoinfer.registry.model_card import class_colors

# pathotrain's export defaults, which its metadata does not state.
RECOMMENDED_OVERLAP = 0.25
DETECTION_SCORE_THRESHOLD = 0.5


def card_fields(meta: dict[str, Any], model_file: Path) -> dict[str, Any]:
    task = meta["task"]
    names = _class_names(task, meta)
    colors = class_colors(names)
    fields: dict[str, Any] = {
        "name": model_file.stem,
        "version": "0",
        "task": task,
        "framework": "onnx",
        "file": model_file.name,
        "precision": "fp32",
        "input": {
            "name": "image",
            "tile_size": meta["tile_size"],
            "channel_order": "RGB",
            "layout": "NCHW",
            "dtype": "float32",
            "normalisation": {"mean": meta["mean"], "std": meta["std"], "scale": 1 / 255},
            "icc": True,
        },
        "classes": [{"id": i, "name": n, "color": colors[n]} for i, n in enumerate(names)],
        "recommended_overlap": RECOMMENDED_OVERLAP,
        "provenance": {"source_metadata": {"pathotrain": meta}},
    }
    if meta.get("tile_mpp") is not None:
        fields["input"]["mpp"] = meta["tile_mpp"]
    if task == "segmentation":
        fields["background_id"] = 0
        fields["output"] = {"tensor": "logits", "activation": "logits", "n_classes": len(names)}
    elif task == "detection":
        fields["output"] = {
            "boxes": "boxes",
            "scores": "scores",
            "labels": "labels",
            "box_format": "xyxy",
            "box_units": "pixel",
            "score_threshold": DETECTION_SCORE_THRESHOLD,
        }
    return fields


def _class_names(task: str, meta: dict[str, Any]) -> list[str]:
    """``class_names`` by output index. pathotrain fills unnamed classes with ``"0"``,
    ``"1"``, ...; those take their name from ``class_map`` (annotation name -> label)
    instead, and segmentation label 0 is background."""
    names = [str(n) for n in meta.get("class_names") or range(meta["num_classes"])]
    by_label = {label: name for name, label in (meta.get("class_map") or {}).items()}
    for i, name in enumerate(names):
        if name == str(i):
            names[i] = by_label.get(i) or (
                "background" if task == "segmentation" and i == 0 else name
            )
    return names
