import json
import shutil
from pathlib import Path

import pytest

from pathoinfer.cli import main
from pathoinfer.registry.loader import load_model

FIXTURES = Path(__file__).parent / "fixtures"
ort = pytest.importorskip("onnxruntime")


def test_card_create_wraps_a_bare_onnx_with_user_fields(tmp_path: Path) -> None:
    fields = tmp_path / "fields.yaml"
    fields.write_text(
        """
task: segmentation
input: {mpp: 0.5, normalisation: {mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]}}
classes:
  - {id: 0, name: background, color: [0, 0, 0]}
  - {id: 1, name: tumor, color: [255, 0, 0]}
background_id: 0
output: {activation: logits, n_classes: 2}
recommended_overlap: 0.25
"""
    )
    out = tmp_path / "wrapped"
    model = FIXTURES / "tiny_seg" / "model.onnx"
    assert (
        main(["card", "create", str(model), str(out), "--fields", str(fields), "--version", "2"])
        == 0
    )
    card, folder = load_model(out)
    assert (card.name, card.version, card.input.tile_size, card.input.name) == (
        "model",
        "2",
        16,
        "image",
    )
    assert main(["card", "validate", str(folder)]) == 0


def test_card_create_prefills_from_the_legacy_pathotrain_key(tmp_path: Path) -> None:
    out = tmp_path / "wrapped"
    legacy = FIXTURES / "tiny_seg_legacy.onnx"
    assert main(["card", "create", str(legacy), str(out), "--name", "tumour-seg"]) == 0
    card = json.loads((out / "model_card.json").read_text())
    assert card["name"] == "tumour-seg" and card["input"]["mpp"] == 0.5
    assert (out / "tiny_seg_legacy.onnx").is_file()


def test_card_create_reports_missing_fields_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "wrapped"
    assert main(["card", "create", str(FIXTURES / "tiny_seg" / "model.onnx"), str(out)]) == 1
    assert "task" in capsys.readouterr().err
    assert not out.exists()


def test_card_create_refuses_a_non_empty_folder(tmp_path: Path) -> None:
    (tmp_path / "keep.txt").write_text("x")
    assert main(["card", "create", str(FIXTURES / "tiny_seg_legacy.onnx"), str(tmp_path)]) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.txt"]


def test_card_validate_fails_on_a_graph_mismatch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shutil.copytree(FIXTURES / "tiny_seg", tmp_path / "m")
    card = json.loads((tmp_path / "m" / "model_card.json").read_text())
    card["input"]["name"] = "pixels"
    (tmp_path / "m" / "model_card.json").write_text(json.dumps(card))
    assert main(["card", "validate", str(tmp_path / "m")]) == 1
    assert "'pixels' is not a graph input" in capsys.readouterr().err
