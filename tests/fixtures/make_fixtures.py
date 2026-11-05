"""Regenerate the committed model fixtures (needs the ``onnx`` package, dev time only)::

    uv run python tests/fixtures/make_fixtures.py

One 2-layer segmentation network (1x1 conv 3->4, ReLU, 1x1 conv 4->2) with fixed weights,
saved as:

tiny_seg/             model folder: model.onnx + model_card.json (hand-written card)
tiny_seg_legacy.onnx  the graph with only the legacy ``pathotrain`` metadata key
tiny_seg_card.onnx    the graph with the card embedded under ``model_card``
tiny_seg_fp16/        the graph in float16, card with float16 input and fp16 precision
tiny_seg_torch/       TorchScript (model.pt)
tiny_seg_export/      torch.export (model.pt2)
"""

import json
from pathlib import Path

import numpy as np
import onnx
import torch
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


def weights() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(size=(4, 3, 1, 1)), rng.normal(size=(2, 4, 1, 1))


def graph(metadata: dict[str, str], fp16: bool = False) -> onnx.ModelProto:
    dtype, elem = (np.float16, TensorProto.FLOAT16) if fp16 else (np.float32, TensorProto.FLOAT)
    w1, w2 = (
        numpy_helper.from_array(w.astype(dtype), n) for w, n in zip(weights(), "12", strict=True)
    )
    w1.name, w2.name = "w1", "w2"
    nodes = [
        helper.make_node("Conv", ["image", "w1"], ["hidden"]),
        helper.make_node("Relu", ["hidden"], ["relu"]),
        helper.make_node("Conv", ["relu", "w2"], ["logits"]),
    ]
    g = helper.make_graph(
        nodes,
        "tiny_seg",
        [helper.make_tensor_value_info("image", elem, ["batch", 3, SIZE, SIZE])],
        [helper.make_tensor_value_info("logits", elem, ["batch", 2, SIZE, SIZE])],
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
    fp16_card = card | {"precision": "fp16", "input": card["input"] | {"dtype": "float16"}}
    write_folder("tiny_seg_fp16", fp16_card, "model.onnx", lambda p: onnx.save(graph({}, True), p))

    net = torch.nn.Sequential(
        torch.nn.Conv2d(3, 4, 1, bias=False), torch.nn.ReLU(), torch.nn.Conv2d(4, 2, 1, bias=False)
    ).eval()
    with torch.no_grad():
        for conv, w in zip((net[0], net[2]), weights(), strict=True):
            conv.weight.copy_(torch.from_numpy(w.astype(np.float32)))
    example = torch.zeros(1, 3, SIZE, SIZE)
    write_folder(
        "tiny_seg_torch",
        card | {"framework": "torch", "file": "model.pt"},
        "model.pt",
        lambda p: torch.jit.save(torch.jit.trace(net, example), p),
    )
    batch = torch.export.Dim("batch")
    program = torch.export.export(net, (example.repeat(2, 1, 1, 1),), dynamic_shapes=({0: batch},))
    write_folder(
        "tiny_seg_export",
        card | {"framework": "torch", "file": "model.pt2"},
        "model.pt2",
        lambda p: torch.export.save(program, p),
    )


def write_folder(name: str, card: dict, file: str, save) -> None:  # type: ignore[no-untyped-def]
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    save(folder / file)
    (folder / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")


if __name__ == "__main__":
    main()
