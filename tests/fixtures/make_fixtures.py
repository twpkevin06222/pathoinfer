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

Plus a detector, ``tiny_det/`` (torch.export): every tile gets one box at its centre
(score 0.9) and one in its corner (score 0.1, below the card's threshold), labelled 1
where the tile is redder than it is blue. So each tile owns exactly one detection.

And ``tiny_dark/`` (torch.export, 64 px tiles): class 1 wherever a pixel is darker than
glass (mean RGB below 0.85), so its polygons trace tissue edges exactly; alignment and
area checks compare them with the shapes drawn on the synthetic slide.

And ``tiny_nuclei/`` (torch.export, 64 px tiles, task ``instance``): HoVer-Net style
outputs where every dark region of a tile is one "nucleus": ``np`` is tiny_dark's logits,
``hv`` the region times an x and a y ramp (rising across it and dropping at its edge, which
seeds the watershed), ``nt`` type 1 where dark. Run only ``nuclei()`` to add it alone.
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
    detector()
    dark()
    dark_onnx()
    nuclei()


class TinyDet(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        b, c = image.shape[0], SIZE / 2
        boxes = torch.tensor([[c - 4, c - 4, c + 4, c + 4], [0.0, 0.0, 4.0, 4.0]])
        labels = (image[:, 0].mean((1, 2)) > image[:, 2].mean((1, 2))).long()
        return {
            "boxes": boxes.expand(b, 2, 4),
            "scores": torch.tensor([0.9, 0.1]).expand(b, 2),
            "labels": labels[:, None].expand(b, 2),
        }


def detector() -> None:
    card = json.loads((HERE / "tiny_seg" / "model_card.json").read_text())
    card = {k: v for k, v in card.items() if k not in ("background_id", "output")} | {
        "name": "tiny-det",
        "task": "detection",
        "framework": "torch",
        "file": "model.pt2",
        "classes": [
            {"id": 0, "name": "lymphocyte", "color": [0, 0, 255]},
            {"id": 1, "name": "tumour", "color": [255, 0, 0]},
        ],
        "output": {"box_format": "xyxy", "box_units": "pixel", "score_threshold": 0.5},
    }
    batch = torch.export.Dim("batch")
    example = torch.zeros(2, 3, SIZE, SIZE)
    program = torch.export.export(TinyDet(), (example,), dynamic_shapes=({0: batch},))
    write_folder("tiny_det", card, "model.pt2", lambda p: torch.export.save(program, p))


class TinyDark(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        darker = 40 * (0.85 - image.mean(1, keepdim=True))
        return torch.cat([-darker, darker], 1)


def dark() -> None:
    card = json.loads((HERE / "tiny_seg" / "model_card.json").read_text())
    card |= {"name": "tiny-dark", "framework": "torch", "file": "model.pt2"}
    card["input"] |= {
        "tile_size": 64,
        "normalisation": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
    }
    batch = torch.export.Dim("batch")
    example = torch.zeros(2, 3, 64, 64)
    program = torch.export.export(TinyDark(), (example,), dynamic_shapes=({0: batch},))
    write_folder("tiny_dark", card, "model.pt2", lambda p: torch.export.save(program, p))


class TinyNuclei(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        darker = 40 * (0.85 - image.mean(1, keepdim=True))
        logits = torch.cat([-darker, darker], 1)
        dark = torch.sigmoid(darker)
        ramp = torch.arange(image.shape[-1], dtype=image.dtype) / image.shape[-1]
        hv = torch.cat([dark * ramp, dark * ramp[:, None]], 1)
        return {"np": logits, "hv": hv, "nt": logits}


def nuclei() -> None:
    card = json.loads((HERE / "tiny_dark" / "model_card.json").read_text())
    card = {k: v for k, v in card.items() if k != "output"} | {
        "name": "tiny-nuclei",
        "task": "instance",
        "classes": [
            {"id": 0, "name": "background", "color": [0, 0, 0]},
            {"id": 1, "name": "nucleus", "color": [200, 0, 200]},
        ],
        "output": {"n_classes": 2},
    }
    batch = torch.export.Dim("batch")
    example = torch.zeros(2, 3, 64, 64)
    program = torch.export.export(TinyNuclei(), (example,), dynamic_shapes=({0: batch},))
    write_folder("tiny_nuclei", card, "model.pt2", lambda p: torch.export.save(program, p))


def dark_onnx() -> None:
    """tiny_dark as ONNX (for Triton and the Docker demo): the same function, written with
    onnx.helper so it needs no exporter."""
    card = json.loads((HERE / "tiny_dark" / "model_card.json").read_text())
    card |= {"name": "tiny-dark-onnx", "framework": "onnx", "file": "model.onnx"}
    constants = [
        numpy_helper.from_array(np.array(0.85, np.float32), "threshold"),
        numpy_helper.from_array(np.array(40.0, np.float32), "gain"),
    ]
    nodes = [
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[1], keepdims=1),
        helper.make_node("Sub", ["threshold", "mean"], ["margin"]),
        helper.make_node("Mul", ["margin", "gain"], ["darker"]),
        helper.make_node("Neg", ["darker"], ["lighter"]),
        helper.make_node("Concat", ["lighter", "darker"], ["logits"], axis=1),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_dark",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 64, 64])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2, 64, 64])],
        constants,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    write_folder("tiny_dark_onnx", card, "model.onnx", lambda p: onnx.save(model, p))


def write_folder(name: str, card: dict, file: str, save) -> None:  # type: ignore[no-untyped-def]
    folder = HERE / name
    folder.mkdir(exist_ok=True)
    save(folder / file)
    (folder / "model_card.json").write_text(json.dumps(card, indent=2) + "\n")


if __name__ == "__main__":
    main()
