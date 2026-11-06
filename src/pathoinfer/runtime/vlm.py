"""What every VLM runtime shares: how an image and a question become a chat message, and
how boxes in an answer become level-0 slide coordinates.

The vLLM server client (``runtime/remote/vllm.py``) and offline generation
(``runtime/local/vllm_offline.py``) build the same OpenAI-style message, so a model
answers the same question the same way through either.

Grounding answers (``VqaOutput.grounding``) hold boxes relative to the image the model
saw: per mille (0-1000, as Qwen3-VL), normalised (0-1) or pixels of that image.
``parse_boxes`` maps them through the ``Crop`` the image was read from to level-0 pixels
(x right, y down), clipped to the crop.
"""

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from pathoinfer.registry.model_card import VqaOutput

Image = str | Path | NDArray[np.uint8]  # an image file, or an RGB array (H, W, 3)

_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")
_BRACKETS = re.compile(r"\[([^\[\]]*)\]")
_BOX_KEYS = ("bbox_2d", "bbox", "box")


def image_url(image: Image, local: bool = False) -> str:
    """The image as a URL an OpenAI-compatible server takes: a ``file://`` URI when
    ``local`` and the image is a file (the server must run with
    ``--allowed-local-media-path`` covering it), else a base64 data URL (arrays as PNG)."""
    if isinstance(image, np.ndarray):
        ok, png = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError(f"cannot encode an image of shape {image.shape}")
        return "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
    path = Path(image)
    if local:
        return path.resolve().as_uri()
    suffix = path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else suffix
    return f"data:image/{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def chat_messages(url: str, question: str, system: str | None = None) -> list[dict[str, Any]]:
    """One user turn: the image, then the question (after an optional system prompt)."""
    content = [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": question}]
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if system:
        messages.insert(0, {"role": "system", "content": system})
    return messages


@dataclass(frozen=True)
class Crop:
    """Where the image a VLM saw came from: its level-0 region (``x``, ``y``, ``width``,
    ``height``) and its size in pixels as sent (``image_size``, width then height). A
    crop of ``tile_size`` px read at ``mpp`` covers ``tile_size * mpp / slide_mpp``
    level-0 px."""

    x: float
    y: float
    width: float
    height: float
    image_size: tuple[int, int]


@dataclass(frozen=True)
class Grounding:
    """Boxes from one answer: ``boxes`` (n, 4) ``xyxy`` level-0 px, and their labels
    (``""`` where the answer gave none)."""

    boxes: NDArray[np.float64]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.labels)


def parse_boxes(answer: str, output: VqaOutput, crop: Crop) -> Grounding:
    """Every box in ``answer`` in level-0 px. Reads JSON first (Qwen-VL's
    ``[{"bbox_2d": [x0, y0, x1, y1], "label": "..."}]``, code fences allowed), else each
    ``[...]`` holding four numbers, else the first four numbers of the text."""
    found = _json_boxes(answer) or _text_boxes(answer)
    if not found:
        return Grounding(np.zeros((0, 4)), [])
    raw = np.array([box for box, _ in found], dtype=np.float64)
    if output.box_format == "cxcywh":
        cx, cy, w, h = raw.T
        raw = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    image_w, image_h = crop.image_size
    unit = {"per_mille": (1000.0, 1000.0), "normalised": (1.0, 1.0), "pixel": (image_w, image_h)}
    ux, uy = unit[output.box_units]
    fx = np.clip(raw[:, [0, 2]] / ux, 0.0, 1.0)  # fraction of the image's width
    fy = np.clip(raw[:, [1, 3]] / uy, 0.0, 1.0)
    xs, ys = crop.x + np.sort(fx, axis=1) * crop.width, crop.y + np.sort(fy, axis=1) * crop.height
    boxes = np.stack([xs[:, 0], ys[:, 0], xs[:, 1], ys[:, 1]], axis=1)
    return Grounding(boxes, [label for _, label in found])


def _json_boxes(answer: str) -> list[tuple[list[float], str]]:
    starts = [i for i in (answer.find("["), answer.find("{")) if i >= 0]
    if not starts:
        return []
    start = min(starts)
    end = max(answer.rfind("]"), answer.rfind("}")) + 1
    try:
        data = json.loads(answer[start:end])
    except ValueError:
        return []
    items = data if isinstance(data, list) else [data]
    found = []
    for item in items:
        if not isinstance(item, dict):
            continue
        box = next((item[k] for k in _BOX_KEYS if k in item), None)
        if isinstance(box, list) and len(box) == 4 and all(_is_number(v) for v in box):
            found.append(([float(v) for v in box], str(item.get("label", ""))))
    return found


def _text_boxes(answer: str) -> list[tuple[list[float], str]]:
    lists = [_NUMBER.findall(inner) for inner in _BRACKETS.findall(answer)]
    boxes = [n[:4] for n in lists if len(n) >= 4]
    if not boxes:
        numbers = _NUMBER.findall(answer)
        boxes = [numbers[:4]] if len(numbers) >= 4 else []
    return [([float(v) for v in box], "") for box in boxes]


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
