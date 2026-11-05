"""Regenerate the committed ONNX fixtures (needs the ``onnx`` package, dev time only)::

    uv run python tests/fixtures/make_fixtures.py

tiny_seg/            model folder: model.onnx + model_card.json
tiny_seg_legacy.onnx the same graph with only the legacy ``pathotrain`` metadata key
tiny_seg_card.onnx   the same graph with the card embedded under ``model_card``
"""

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).parent
SIZE = 16
LEGACY = {
    "task": "segmentation",
    "model": "smp:unet",
    "input": f"image: (batch, 3, {SIZE}, {SIZE}) float32 RGB; tile / 255, then (x - mean) / std",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "tile_size": SIZE,
    "tile_mpp": 0.5,
    "num_classes": 2,
    "class_names": ["0", "1"],
    "class_map": {"tumor": 1},
    "outputs": {"logits": "(batch, num_classes, size, size) float32; argmax over classes"},
}


def graph(metadata: dict[str, str]) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    w1 = numpy_helper.from_array(rng.normal(size=(4, 3, 1, 1)).astype(np.float32), "w1")
    w2 = numpy_helper.from_array(rng.normal(size=(2, 4, 1, 1)).astype(np.float32), "w2")
    nodes = [
        helper.make_node("Conv", ["image", "w1"], ["hidden"]),
        helper.make_node("Relu", ["hidden"], ["relu"]),
        helper.make_node("Conv", ["relu", "w2"], ["logits"]),
    ]
    g = helper.make_graph(
        nodes,
        "tiny_seg",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2, SIZE, SIZE])],
        [w1, w2],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    for key, value in metadata.items():
        model.metadata_props.add(key=key, value=value)
    onnx.checker.check_model(model)
    return model


def main() -> None:
    card = json.loads((HERE / "tiny_seg" / "model_card.json").read_text())
    onnx.save(graph({}), HERE / "tiny_seg" / "model.onnx")
    onnx.save(graph({"pathotrain": json.dumps(LEGACY)}), HERE / "tiny_seg_legacy.onnx")
    onnx.save(
        graph({"model_card": json.dumps(card | {"file": "tiny_seg_card.onnx"})}),
        HERE / "tiny_seg_card.onnx",
    )


if __name__ == "__main__":
    main()
