"""Conversion and the parity gate on the tiny fixture models and a synthetic slide."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from pathoinfer.cli import main
from pathoinfer.config import CalibrationConfig, ParityConfig, QuantConfig
from pathoinfer.convert import parity
from pathoinfer.convert.common import match_nodes, write_variant
from pathoinfer.convert.fp16 import to_fp16
from pathoinfer.convert.parity import auc, gate
from pathoinfer.registry.loader import load_model

onnx = pytest.importorskip("onnx")
FIXTURES = Path(__file__).parents[1] / "fixtures"
TINY = FIXTURES / "tiny_seg"


def parity_config(slide: Path, **options: Any) -> ParityConfig:
    return ParityConfig.model_validate(
        {
            "slides": str(slide),
            "window_tiles": 6,
            "windows": 2,
            "reference_device": "cpu",
            "candidate": {"device": "cpu", "batch_size": 16},
            "segmentation": {"output_mpp": 0.5, "simplify_px": 0, "smooth": 0},
        }
        | options
    )


def broken(out: Path) -> Path:
    """tiny_seg with its last layer negated: the classes swap, a broken conversion."""
    card, folder = load_model(str(TINY))
    model = onnx.load(folder / card.file)
    w2 = next(t for t in model.graph.initializer if t.name == "w2")
    flipped = -onnx.numpy_helper.to_array(w2)
    w2.CopyFrom(onnx.numpy_helper.from_array(flipped, "w2"))
    write_variant(card, out, lambda p: onnx.save(model, p), precision="fp16", version="1+bad")
    return out


def test_fp16_conversion_records_its_parent(tmp_path: Path) -> None:
    card = to_fp16(TINY, tmp_path / "fp16")
    assert card.precision == "fp16" and card.version == "1+fp16"
    assert card.provenance.parent_model == "tiny-seg@1"
    assert card.input.dtype == "float32"  # inputs and outputs kept float32
    model = onnx.load(tmp_path / "fp16" / "model.onnx")
    types = {t.data_type for t in model.graph.initializer}
    assert types == {onnx.TensorProto.FLOAT16}
    with pytest.raises(FileExistsError):
        to_fp16(TINY, tmp_path / "fp16")


def test_fp16_keeps_external_weights_beside_the_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Models over 2 GB keep weights in ``model.onnx.data``: the conversion reads them
    from beside the graph and, still large in FP16, writes them the same way."""
    import shutil

    from pathoinfer.convert import fp16
    from pathoinfer.runtime.local.onnx import OnnxRuntime

    source = tmp_path / "big"
    shutil.copytree(TINY, source)
    onnx.save(
        onnx.load(source / "model.onnx"),
        source / "model.onnx",
        save_as_external_data=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    monkeypatch.setattr(fp16, "EXTERNAL_BYTES", 0)  # the tiny model counts as large
    card = fp16.to_fp16(source, tmp_path / "fp16")
    assert (tmp_path / "fp16" / "model.onnx.data").is_file()
    side = card.input.tile_size
    tiles = torch.randint(0, 256, (2, side, side, 3), dtype=torch.uint8)
    reference, _ = load_model(str(TINY))
    want = OnnxRuntime(reference, TINY / "model.onnx", device="cpu").predict(tiles)
    got = OnnxRuntime(card, tmp_path / "fp16" / "model.onnx", device="cpu").predict(tiles)
    for name, value in want.items():
        assert np.allclose(np.asarray(got[name]), np.asarray(value), atol=0.05)


def test_fp16_io_changes_the_input_dtype(tmp_path: Path) -> None:
    card = to_fp16(TINY, tmp_path / "fp16", keep_io_types=False)
    assert card.input.dtype == "float16"


def test_fp16_passes_parity_and_writes_the_record(tmp_path: Path, slide: Path) -> None:
    to_fp16(TINY, tmp_path / "fp16")
    report = gate(TINY, tmp_path / "fp16", parity_config(slide), tmp_path / "final")
    assert report.passed, report.summary()
    assert report.tiles > 20 and report.metrics["dice_min"] > 0.99
    card, _ = load_model(str(tmp_path / "final"))
    record = card.provenance.parity
    assert record is not None and record.reference == "tiny-seg@1"
    assert record.metrics == report.metrics and record.tile_set_hash == report.tile_set_hash
    assert card.precision == "fp16" and card.provenance.parent_model == "tiny-seg@1"


def test_broken_conversion_fails_and_names_classes_and_layers(
    tmp_path: Path, slide: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(parity, "register_model", lambda *a, **k: pytest.fail("registered"))
    report = gate(
        TINY,
        broken(tmp_path / "bad"),
        parity_config(slide),
        tmp_path / "final",
        register=("tiny", "fp16-test"),
    )
    assert not report.passed
    assert report.failures and report.metrics["dice_min"] < 0.5
    assert "tumor" in report.per_class["dice_min"]
    assert report.worst_layers[0][0] == "logits" and report.worst_layers[0][1] > 1
    assert not (tmp_path / "final").exists()
    assert "FAIL" in report.summary()


def test_parity_is_reproducible(tmp_path: Path, slide: Path) -> None:
    to_fp16(TINY, tmp_path / "fp16")
    a = parity.check(TINY, tmp_path / "fp16", parity_config(slide))
    b = parity.check(TINY, tmp_path / "fp16", parity_config(slide))
    assert a.tile_set_hash == b.tile_set_hash and a.metrics == b.metrics
    c = parity.check(TINY, tmp_path / "fp16", parity_config(slide, seed=1))
    assert c.tile_set_hash != a.tile_set_hash


def test_ort_int8_quantises_and_records_calibration(tmp_path: Path, slide: Path) -> None:
    cfg = QuantConfig(
        engine="ort",
        calibration=CalibrationConfig(slides=str(slide), tiles=32),
        keep_ends=False,
    )
    from pathoinfer.convert.ptq import quantise

    card = quantise(TINY, tmp_path / "int8", cfg)
    assert card.precision == "int8" and card.provenance.calibration_set_hash
    ops = {n.op_type for n in onnx.load(tmp_path / "int8" / "model.onnx").graph.node}
    assert {"QuantizeLinear", "DequantizeLinear"} <= ops
    again = quantise(TINY, tmp_path / "again", cfg)
    assert again.provenance.calibration_set_hash == card.provenance.calibration_set_hash
    settings = card.provenance.source_metadata["quantisation"]
    assert settings["engine"] == "ort" and settings["tiles"] == 32
    # The quantised model still runs and stays close on this tiny network.
    report = parity.check(TINY, tmp_path / "int8", parity_config(slide))
    assert report.metrics["dice_min"] > 0.9


def test_keep_ends_excludes_the_first_and_last_compute_layers() -> None:
    nodes = [
        ("conv1", "Conv"),
        ("relu", "Relu"),
        ("ln", "LayerNormalization"),
        ("blocks.0.attn.qkv", "MatMul"),
        ("conv2", "Conv"),
    ]
    kept = match_nodes(nodes, ["LayerNormalization"], ["*attn*"], ends=True)
    assert kept == ["conv1", "ln", "blocks.0.attn.qkv", "conv2"]
    assert match_nodes(nodes, [], [], ends=False) == []


def test_fp8_needs_modelopt() -> None:
    with pytest.raises(ValueError, match="modelopt"):
        QuantConfig(precision="fp8", engine="ort", calibration=CalibrationConfig(slides="s"))


@pytest.mark.parametrize(
    ("labels", "scores", "expected"),
    [
        ([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1], 1.0),
        ([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9], 0.0),
        ([1, 0], [0.5, 0.5], 0.5),
        ([1, 1], [0.3, 0.6], 1.0),  # no negatives: nothing can be ranked wrongly
    ],
)
def test_auc(labels: list[int], scores: list[float], expected: float) -> None:
    assert auc(np.array(labels, bool), np.array(scores)) == pytest.approx(expected)


def test_detection_matching_is_one_to_one() -> None:
    ref = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float)
    cand = np.array([[1, 1, 11, 11], [0, 0, 10, 10], [50, 50, 60, 60]], float)
    assert parity._matches(ref, cand, 0.5) == 1


class Fake:
    """A runtime that returns fixed outputs per tile position."""

    def __init__(self, fn: Any, batch_size: int = 9) -> None:
        self.fn, self.batch_size = fn, batch_size
        self.specs: list[Any] = []

    def predict(self, batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.fn(batch)


def test_detection_and_classification_metrics_through_the_stitchers() -> None:
    from tests.stitch import test_detection as det
    from tests.stitch import test_heatmap as hm
    from tests.stitch.test_segmentation import grid

    slide = grid()
    images = np.zeros((len(slide.tiles), 1, 1, 3), np.uint8)
    cfg = ParityConfig(slides="unused", iou=0.5)
    objects = det.squares(np.array([[1000.0, 1000.0], [2000.0, 2000.0], [3000.0, 1500.0]]), 40)
    card = det.card()
    specs = list(slide.tiles)
    labels = np.array([0, 1, 1])

    def detector(drop_last: bool) -> Any:
        keep = objects[:2] if drop_last else objects
        state = {"i": 0}

        def fn(batch: torch.Tensor) -> dict[str, torch.Tensor]:
            chunk = specs[state["i"] : state["i"] + len(batch)]
            state["i"] = (state["i"] + len(batch)) % len(specs)
            return det.detect(chunk, keep, labels[: len(keep)], card)

        return fn

    per_class, _ = parity.compare(
        card, Fake(detector(False)), Fake(detector(True)), [(slide, images)], cfg
    )
    assert per_class["f1_min"]["tumour"] == 1.0
    assert per_class["f1_min"]["lymphocyte"] == pytest.approx(2 * 1 / 3)
    assert per_class["count_diff_max"]["lymphocyte"] == pytest.approx(0.5)

    cls_card = hm.card()

    def classifier(swap: bool) -> Any:
        state = {"i": 0}

        def fn(batch: torch.Tensor) -> dict[str, torch.Tensor]:
            chunk = specs[state["i"] : state["i"] + len(batch)]
            state["i"] = (state["i"] + len(batch)) % len(specs)
            p = torch.tensor([hm.checkerboard(s) for s in chunk])
            p = 1 - p if swap else p
            return {"scores": torch.stack([1 - p, p], dim=1)}

        return fn

    same, _ = parity.compare(
        cls_card, Fake(classifier(False)), Fake(classifier(False)), [(slide, images)], cfg
    )
    swapped, _ = parity.compare(
        cls_card, Fake(classifier(False)), Fake(classifier(True)), [(slide, images)], cfg
    )
    assert same["auc_min"]["tumour"] == 1.0 and swapped["auc_min"]["tumour"] == 0.0


def test_cli_converts_and_gates(tmp_path: Path, slide: Path) -> None:
    config = tmp_path / "parity.yaml"
    config.write_text(json.dumps(parity_config(slide).model_dump(mode="json")))
    assert main(["convert", "fp16", str(TINY), str(tmp_path / "fp16")]) == 0
    args = ["convert", "parity", str(TINY), str(tmp_path / "fp16"), str(tmp_path / "final")]
    assert main([*args, "--config", str(config), "--report", str(tmp_path / "r.json")]) == 0
    assert json.loads((tmp_path / "r.json").read_text())["passed"] is True
    bad = ["convert", "parity", str(TINY), str(broken(tmp_path / "bad")), str(tmp_path / "x")]
    assert main([*bad, "--config", str(config)]) == 1
    assert main([*args[:4], str(tmp_path / "y"), "--config", str(config), "--register", "n"]) == 1


def test_mlflow_registration_with_an_alias(
    tmp_path: Path, slide: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mlflow = pytest.importorskip("mlflow")
    if importlib.util.find_spec("mlflow.store.db") is None:
        pytest.skip("a local sqlite registry needs full mlflow; mlflow-skinny talks to a server")
    monkeypatch.chdir(tmp_path)  # MLflow puts default artifact roots in the working dir
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    to_fp16(TINY, tmp_path / "fp16")
    gate(
        TINY,
        tmp_path / "fp16",
        parity_config(slide),
        tmp_path / "final",
        register=("tiny-seg", "fp16-test"),
        tracking_uri=uri,
    )
    card, _ = load_model("models:/tiny-seg@fp16-test", cache_dir=tmp_path / "cache")
    assert card.provenance.parity is not None
