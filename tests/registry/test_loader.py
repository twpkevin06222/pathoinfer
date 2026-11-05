import json
import shutil
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path

import pytest

from pathoinfer.registry import legacy
from pathoinfer.registry.loader import CardError, graph_problems, load_model
from pathoinfer.registry.model_card import SegmentationCard

FIXTURES = Path(__file__).parents[1] / "fixtures"
FOLDER = FIXTURES / "tiny_seg"


def _installed(dist: str) -> bool:
    try:
        distribution(dist)
    except PackageNotFoundError:
        return False
    return True


def test_folder_loads() -> None:
    card, folder = load_model(FOLDER)
    assert (card.name, card.task, folder) == ("tiny-seg", "segmentation", FOLDER)
    assert (folder / card.file).is_file()


def test_folder_whose_model_file_is_missing_is_rejected(tmp_path: Path) -> None:
    shutil.copy(FOLDER / "model_card.json", tmp_path)
    with pytest.raises(CardError, match="'model.onnx', which is not in"):
        load_model(tmp_path)


def test_folder_without_card_or_model_suggests_card_create(tmp_path: Path) -> None:
    with pytest.raises(CardError, match="pathoinfer card create"):
        load_model(tmp_path)


def test_bare_onnx_with_embedded_card_loads() -> None:
    pytest.importorskip("onnxruntime")
    card, folder = load_model(FIXTURES / "tiny_seg_card.onnx")
    assert (card.name, card.file, folder) == ("tiny-seg", "tiny_seg_card.onnx", FIXTURES)


def test_legacy_onnx_is_adapted_with_a_deprecation_warning() -> None:
    pytest.importorskip("onnxruntime")
    with pytest.warns(DeprecationWarning, match="legacy 'pathotrain'"):
        card, _ = load_model(FIXTURES / "tiny_seg_legacy.onnx")
    assert isinstance(card, SegmentationCard)
    assert [c.name for c in card.classes] == ["background", "tumor"]
    assert (card.background_id, card.input.mpp, card.input.tile_size) == (0, 0.5, 16)
    assert card.provenance.source_metadata["pathotrain"]["model"] == "smp:unet"


def test_legacy_metadata_without_tile_mpp_leaves_mpp_unset() -> None:
    meta = {
        "task": "segmentation",
        "tile_size": 16,
        "tile_mpp": None,
        "num_classes": 2,
        "mean": [0.5] * 3,
        "std": [0.5] * 3,
        "class_names": ["0", "1"],
        "class_map": None,
    }
    fields = legacy.card_fields(meta, Path("m.onnx"))
    assert "mpp" not in fields["input"]
    assert [c["name"] for c in fields["classes"]] == ["background", "1"]


def test_graph_matches_the_fixture_card() -> None:
    pytest.importorskip("onnxruntime")
    card, folder = load_model(FOLDER)
    assert graph_problems(card, folder) == []


def test_graph_mismatch_is_reported(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    shutil.copy(FOLDER / "model.onnx", tmp_path)
    data = json.loads((FOLDER / "model_card.json").read_text())
    data["input"]["tile_size"] = 32
    data["output"]["tensor"] = "probs"
    (tmp_path / "model_card.json").write_text(json.dumps(data))
    card, _ = load_model(tmp_path)
    problems = graph_problems(card, tmp_path)
    assert any("32 px" in p for p in problems) and any("'probs'" in p for p in problems)


def test_mlflow_alias_and_version_resolve_to_the_same_folder(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    if not _installed("mlflow"):
        pytest.skip("a local sqlite registry needs full mlflow; mlflow-skinny talks to a server")
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_registry_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.create_experiment("tiny", artifact_location=(tmp_path / "art").as_uri())
    with mlflow.start_run(experiment_id=experiment) as run:
        mlflow.log_artifacts(str(FOLDER), artifact_path="model")
    client = mlflow.MlflowClient()
    client.create_registered_model("tiny")
    version = client.create_model_version(
        "tiny", source=f"{run.info.artifact_uri}/model", run_id=run.info.run_id
    ).version
    client.set_registered_model_alias("tiny", "champion", version)
    for uri in ("models:/tiny@champion", f"models:/tiny@{version}", f"models:/tiny/{version}"):
        card, folder = load_model(uri, cache_dir=tmp_path / "cache")
        assert card.name == "tiny-seg" and (folder / "model.onnx").is_file(), uri


def test_hf_source_resolves_through_snapshot_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hub = pytest.importorskip("huggingface_hub")
    calls = []

    def snapshot_download(repo_id: str, revision: str | None) -> str:
        calls.append((repo_id, revision))
        return str(FOLDER)

    monkeypatch.setattr(hub, "snapshot_download", snapshot_download)
    card, folder = load_model("hf://org/tiny-seg@main")
    assert (card.name, folder, calls) == ("tiny-seg", FOLDER, [("org/tiny-seg", "main")])
