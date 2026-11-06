"""ModelOpt and TensorRT paths: skipped without those packages; plans also need a GPU."""

from pathlib import Path

import pytest

from pathoinfer.config import CalibrationConfig, QuantConfig
from pathoinfer.convert import parity
from pathoinfer.convert.fp16 import to_fp16
from pathoinfer.registry.loader import load_model

from .test_convert import TINY, parity_config

onnx = pytest.importorskip("onnx")


def wide(out: Path) -> Path:
    """A segmentation folder ModelOpt quantises: 3x3 convs with 32 channels (it leaves
    convs with a handful of channels, like tiny_seg's, in high precision)."""
    import json

    import numpy as np
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(0)
    shapes = {"w1": (32, 3, 3, 3), "w2": (32, 32, 3, 3), "w3": (2, 32, 1, 1)}
    weights = [
        numpy_helper.from_array(
            (rng.standard_normal(s) / np.sqrt(np.prod(s[1:]))).astype(np.float32), n
        )
        for n, s in shapes.items()
    ]
    pads = {"pads": [1, 1, 1, 1]}
    nodes = [
        helper.make_node("Conv", ["image", "w1"], ["h1"], name="conv1", **pads),
        helper.make_node("Relu", ["h1"], ["r1"], name="relu1"),
        helper.make_node("Conv", ["r1", "w2"], ["h2"], name="conv2", **pads),
        helper.make_node("Relu", ["h2"], ["r2"], name="relu2"),
        helper.make_node("Conv", ["r2", "w3"], ["logits"], name="conv3"),
    ]
    graph = helper.make_graph(
        nodes,
        "wide",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 3, 16, 16])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 2, 16, 16])],
        weights,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    out.mkdir()
    onnx.save(model, out / "model.onnx")
    card = json.loads((TINY / "model_card.json").read_text()) | {"name": "wide-seg"}
    (out / "model_card.json").write_text(json.dumps(card))
    return out


def quant(slide: Path, precision: str = "int8") -> QuantConfig:
    return QuantConfig(
        engine="modelopt",
        precision=precision,  # type: ignore[arg-type]
        calibration=CalibrationConfig(slides=str(slide), tiles=32),
        keep_ends=False,
    )


def test_modelopt_int8_writes_qdq_and_records_calibration(tmp_path: Path, slide: Path) -> None:
    pytest.importorskip("modelopt.onnx.quantization")
    from pathoinfer.convert.ptq import quantise

    card = quantise(wide(tmp_path / "wide"), tmp_path / "int8", quant(slide))
    assert card.precision == "int8" and card.provenance.calibration_set_hash
    ops = {n.op_type for n in onnx.load(tmp_path / "int8" / "model.onnx").graph.node}
    assert "QuantizeLinear" in ops and "DequantizeLinear" in ops


@pytest.mark.gpu
def test_fp16_plan_builds_caches_and_passes_parity(tmp_path: Path, slide: Path) -> None:
    pytest.importorskip("tensorrt")
    from pathoinfer.convert.tensorrt import build_plan

    to_fp16(TINY, tmp_path / "fp16")
    cache = tmp_path / "cache"
    card = build_plan(tmp_path / "fp16", tmp_path / "plan", max_batch=32, cache=cache)
    assert card.framework == "tensorrt" and card.file == "model.plan"
    assert card.precision == "fp16" and "+trt-" in card.version
    meta = card.provenance.source_metadata["tensorrt"]
    assert meta["compute_capability"] and meta["tensorrt"]
    cached = sorted(cache.glob("*.plan"))
    build_plan(tmp_path / "fp16", tmp_path / "plan2", max_batch=32, cache=cache)
    assert sorted(cache.glob("*.plan")) == cached  # second build came from the cache
    cfg = parity_config(slide, candidate={"device": "cuda", "batch_size": 16})
    report = parity.gate(TINY, tmp_path / "plan", cfg, tmp_path / "final")
    assert report.passed, report.summary()
    final, _ = load_model(str(tmp_path / "final"))
    assert final.provenance.parity is not None


@pytest.mark.gpu
@pytest.mark.parametrize("precision", ["int8", "fp8"])
def test_modelopt_plans_run_through_parity(tmp_path: Path, slide: Path, precision: str) -> None:
    pytest.importorskip("modelopt.onnx.quantization")
    pytest.importorskip("tensorrt")
    from pathoinfer.convert.ptq import quantise
    from pathoinfer.convert.tensorrt import build_plan

    reference = wide(tmp_path / "wide")
    quantise(reference, tmp_path / "q", quant(slide, precision))
    graph = onnx.load(tmp_path / "q" / "model.onnx").graph
    assert "QuantizeLinear" in {n.op_type for n in graph.node}
    card = build_plan(tmp_path / "q", tmp_path / "plan", max_batch=32, cache=tmp_path / "cache")
    assert card.precision == precision
    cfg = parity_config(slide, candidate={"device": "cuda", "batch_size": 16})
    report = parity.check(reference, tmp_path / "plan", cfg)
    assert report.tiles > 20 and "dice_min" in report.metrics


def test_ort_quantisation_still_works_after_modelopt(tmp_path: Path, slide: Path) -> None:
    """ModelOpt patches ONNX Runtime's calibrators in the process it runs in; it runs in
    a child process, so ORT entropy calibration keeps working here."""
    pytest.importorskip("modelopt.onnx.quantization")
    from pathoinfer.convert.ptq import quantise

    reference = wide(tmp_path / "wide")
    quantise(reference, tmp_path / "mo", quant(slide))
    ort = quant(slide).model_copy(update={"engine": "ort", "method": "entropy"})
    assert quantise(reference, tmp_path / "ort", ort).precision == "int8"
