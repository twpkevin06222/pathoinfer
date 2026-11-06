import json
import shlex
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from pathoinfer import cli
from pathoinfer.registry.loader import load_vlm
from pathoinfer.serving.vllm import VllmServeConfig, engine_args, serve_command

COMPRESSED = {"quant_method": "compressed-tensors", "format": "float-quantized"}


def test_bf16_card_serves_in_bfloat16_with_one_image_per_prompt(
    vqa_folder: Callable[..., Path],
) -> None:
    card, folder = load_vlm(vqa_folder())
    assert engine_args(card, folder, VllmServeConfig()) == {
        "model": str(folder),
        "served_model_name": "path-vqa",
        "dtype": "bfloat16",
        "max_model_len": 8192,
        "gpu_memory_utilization": 0.9,
        "tensor_parallel_size": 1,
        "limit_mm_per_prompt": {"image": 1},
    }


def test_quantised_card_takes_the_folders_quantisation(vqa_folder: Callable[..., Path]) -> None:
    card, folder = load_vlm(vqa_folder("fp8", COMPRESSED))
    args = engine_args(card, folder, VllmServeConfig())
    assert args["quantization"] == "compressed-tensors" and args["dtype"] == "bfloat16"


@pytest.mark.parametrize(("precision", "quant"), [("fp8", None), ("bf16", COMPRESSED)])
def test_card_and_weights_must_agree(
    vqa_folder: Callable[..., Path], precision: str, quant: dict[str, str] | None
) -> None:
    card, folder = load_vlm(vqa_folder(precision, quant))
    with pytest.raises(ValueError, match="config.json"):
        engine_args(card, folder, VllmServeConfig())


def test_serve_command_carries_sizing_media_path_and_max_pixels(
    vqa_folder: Callable[..., Path], tmp_path: Path
) -> None:
    card, folder = load_vlm(vqa_folder("fp16", max_pixels=200704))
    cfg = VllmServeConfig(
        port=8001, max_num_seqs=16, gpu_memory_utilization=0.5, allowed_local_media_path=tmp_path
    )
    command = serve_command(card, folder, cfg)
    assert command[:7] == ["vllm", "serve", str(folder), "--host", "0.0.0.0", "--port", "8001"]
    flags = dict(zip(command[7::2], command[8::2], strict=True))
    assert flags["--dtype"] == "float16" and flags["--gpu-memory-utilization"] == "0.5"
    assert flags["--max-num-seqs"] == "16"
    assert flags["--allowed-local-media-path"] == str(tmp_path.resolve())
    assert json.loads(flags["--mm-processor-kwargs"]) == {"max_pixels": 200704}
    assert json.loads(flags["--limit-mm-per-prompt"]) == {"image": 1}


def test_cli_prints_the_command(
    vqa_folder: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = vqa_folder()
    config = tmp_path / "vllm.yaml"
    config.write_text("max_model_len: 4096\nport: 9000\n")
    assert cli.main(["serve", "vllm", str(folder), "--config", str(config)]) == 0
    card, model_dir = load_vlm(folder)
    expected = serve_command(card, model_dir, VllmServeConfig(max_model_len=4096, port=9000))
    assert shlex.split(capsys.readouterr().out) == expected


def test_arguments_are_vllms_own(vqa_folder: Callable[..., Path], tmp_path: Path) -> None:
    """In a venv with vLLM: every engine argument is an ``EngineArgs`` field and every
    flag is one ``vllm serve`` accepts."""
    vllm = pytest.importorskip("vllm")
    import dataclasses

    card, folder = load_vlm(vqa_folder("fp8", COMPRESSED, max_pixels=200704))
    cfg = VllmServeConfig(max_num_seqs=16, allowed_local_media_path=tmp_path)
    fields = {f.name for f in dataclasses.fields(vllm.EngineArgs)}
    assert set(engine_args(card, folder, cfg)) <= fields
    exe = Path(sys.executable).parent / "vllm"  # the venv's own, on PATH or not
    help_text = subprocess.run(
        [str(exe), "serve", "--help=all"], capture_output=True, text=True, check=True
    ).stdout
    flags = [a for a in serve_command(card, folder, cfg) if a.startswith("--")]
    assert [f for f in flags if f not in help_text] == []
