"""Read an ONNX file's metadata and tensor signature through onnxruntime ([onnx] extra)."""

from dataclasses import dataclass
from pathlib import Path

from pathoinfer._extras import extra, require

Shape = list[int | str | None]


@dataclass(frozen=True)
class OnnxInfo:
    metadata: dict[str, str]
    inputs: dict[str, Shape]
    outputs: dict[str, Shape]
    output_types: dict[str, str]  # onnxruntime type strings, e.g. "tensor(float)"


def read_onnx(path: Path) -> OnnxInfo:
    ort = require("onnxruntime", extra("onnx"))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return OnnxInfo(
        metadata=dict(session.get_modelmeta().custom_metadata_map),
        inputs={i.name: list(i.shape) for i in session.get_inputs()},
        outputs={o.name: list(o.shape) for o in session.get_outputs()},
        output_types={o.name: o.type for o in session.get_outputs()},
    )
