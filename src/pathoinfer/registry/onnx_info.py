"""Read an ONNX file's metadata and tensor signature through onnxruntime, and the
external weight files it references through onnx (both in the [onnx] extra)."""

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


def external_data_files(path: Path) -> list[str]:
    """The files, relative to the graph's folder, that hold ``path``'s external weights
    (graphs over 2 GB keep them outside the protobuf); empty for a self-contained graph.
    The names are the graph's own, so copies must keep them beside the graph."""
    onnx = require("onnx", extra("onnx"))
    graph = onnx.load(str(path), load_external_data=False).graph
    tensors = [*graph.initializer, *(a.t for n in graph.node for a in n.attribute if a.t.name)]
    locations = {
        entry.value
        for tensor in tensors
        if tensor.data_location == onnx.TensorProto.EXTERNAL
        for entry in tensor.external_data
        if entry.key == "location"
    }
    return sorted(locations)
