"""Triton: the repository generator (validated against Triton's own config schema) and,
when a server is reachable, TritonRuntime against the local ONNX runtime."""

import os
from pathlib import Path
from typing import Any

import pytest
import torch

from pathoinfer.config import RunConfig, RuntimeConfig
from pathoinfer.registry.loader import load_model
from pathoinfer.runtime.factory import make_runtime

pb = pytest.importorskip("tritonclient.grpc.model_config_pb2")
text_format = pytest.importorskip("google.protobuf.text_format")
from pathoinfer.serving.triton_repo import TritonRepoConfig, write_repository

FIXTURES = Path(__file__).parents[1] / "fixtures"


def parse(path: Path) -> Any:
    return text_format.Parse(path.read_text(), pb.ModelConfig())


def test_onnx_model_becomes_a_valid_triton_model(tmp_path: Path) -> None:
    (model,) = write_repository([FIXTURES / "tiny_dark_onnx"], tmp_path / "repo")
    assert model == tmp_path / "repo" / "tiny-dark-onnx"
    assert (model / "1" / "model.onnx").read_bytes() == (
        FIXTURES / "tiny_dark_onnx" / "model.onnx"
    ).read_bytes()
    config = parse(model / "config.pbtxt")
    assert config.name == "tiny-dark-onnx" and config.platform == "onnxruntime_onnx"
    assert config.max_batch_size == 32
    (inp,) = config.input
    assert (inp.name, pb.DataType.Name(inp.data_type), list(inp.dims)) == (
        "image",
        "TYPE_FP32",
        [3, 64, 64],
    )
    (out,) = config.output
    assert (out.name, pb.DataType.Name(out.data_type), list(out.dims)) == (
        "logits",
        "TYPE_FP32",
        [2, 64, 64],
    )
    assert list(config.dynamic_batching.preferred_batch_size) == [8, 16, 32]
    assert config.instance_group[0].count == 1
    assert (model / "model_card.json").is_file()


def test_settings_and_replacement(tmp_path: Path) -> None:
    cfg = TritonRepoConfig(max_batch_size=16, preferred_batch_sizes=[16], instances=2, kind="cpu")
    write_repository([FIXTURES / "tiny_seg"], tmp_path / "repo", cfg)
    (stale := tmp_path / "repo" / "tiny-seg" / "stale.txt").write_text("x")
    write_repository([FIXTURES / "tiny_seg"], tmp_path / "repo", cfg)
    assert not stale.exists()
    config = parse(tmp_path / "repo" / "tiny-seg" / "config.pbtxt")
    assert config.max_batch_size == 16 and config.instance_group[0].count == 2
    assert pb.ModelInstanceGroup.Kind.Name(config.instance_group[0].kind) == "KIND_CPU"
    with pytest.raises(ValueError, match="exceed max_batch_size"):
        TritonRepoConfig(max_batch_size=8, preferred_batch_sizes=[16])


def test_plans_leave_outputs_to_triton(tmp_path: Path) -> None:
    card, _ = load_model(FIXTURES / "tiny_seg")
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "model.plan").write_bytes(b"not a real plan")
    (plan / "model_card.json").write_text(
        card.model_copy(update={"framework": "tensorrt", "file": "model.plan"}).model_dump_json()
    )
    (model,) = write_repository([plan], tmp_path / "repo")
    config = parse(model / "config.pbtxt")
    assert config.platform == "tensorrt_plan" and not config.output
    assert (model / "1" / "model.plan").is_file()


def test_torch_models_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ONNX models and TensorRT plans"):
        write_repository([FIXTURES / "tiny_dark"], tmp_path / "repo")


def triton_url() -> str:
    url = os.environ.get("PATHOINFER_TRITON_URL")
    if not url:
        pytest.skip("set PATHOINFER_TRITON_URL to a Triton serving tiny_dark_onnx")
    return url


@pytest.mark.gpu
@pytest.mark.slow
def test_triton_runtime_matches_the_local_runtime() -> None:
    card, folder = load_model(FIXTURES / "tiny_dark_onnx")
    remote = make_runtime(card, folder, RuntimeConfig(triton_url=triton_url(), batch_size=8))
    local = make_runtime(card, folder, RuntimeConfig(device="cpu", batch_size=8))
    tiles = torch.randint(0, 255, (11, 64, 64, 3), dtype=torch.uint8)
    torch.testing.assert_close(remote.predict(tiles)["logits"], local.predict(tiles)["logits"])


@pytest.mark.gpu
@pytest.mark.slow
def test_the_engine_runs_unchanged_against_triton(tmp_path: Path) -> None:
    from pathoinfer.engine.pipeline import run_batch
    from pathoinfer.tiling.synthetic import write_pyramid

    slide = write_pyramid(tmp_path / "slide_name-01.tiff")
    results = {}
    for name, runtime in [("local", {"device": "cpu"}), ("triton", {"triton_url": triton_url()})]:
        cfg = RunConfig.model_validate(
            {
                "task": "segmentation",
                "model": str(FIXTURES / "tiny_dark_onnx"),
                "slides": str(slide),
                "runtime": runtime | {"batch_size": 16},
                "loader": {"num_workers": 0},
                "outputs": {"dir": str(tmp_path / name)},
            }
        )
        (result,) = run_batch(cfg)
        results[name] = (result.status, result.tiles, result.objects)
    assert results["local"] == results["triton"] and results["local"][0] == "done"
