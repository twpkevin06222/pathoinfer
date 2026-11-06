"""Full VLM runs: quantise a real model on the GPU and gate it through vLLM servers.

Skipped unless the environment names them:

- ``PATHOINFER_VLM_MODEL``: a bf16 VLM folder with a VqaCard (e.g. a merged Qwen3-VL);
- ``PATHOINFER_VLM_PAIRS``: a ``.jsonl``/``.csv``/``.parquet`` of pathology image +
  question (+ answer) rows for calibration and evaluation;
- ``PATHOINFER_VLLM_REFERENCE_URL`` and ``PATHOINFER_VLLM_CANDIDATE_URL``: vLLM servers
  already serving the bf16 model and a quantised one (``vllm serve FOLDER``), for parity.
"""

import os
from pathlib import Path

import pytest

from pathoinfer.config import VlmParityConfig, VlmQuantConfig, VqaDataConfig
from pathoinfer.registry.loader import load_vlm

pytestmark = [pytest.mark.gpu, pytest.mark.slow]


def env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run")
    return value


@pytest.mark.parametrize("scheme", ["fp8", "w4a16-gptq", "w4a16-awq"])
def test_quantise_a_real_vlm(tmp_path: Path, scheme: str) -> None:
    pytest.importorskip("llmcompressor")
    from pathoinfer.convert.vlm import quantise_vlm

    model = Path(env("PATHOINFER_VLM_MODEL"))
    pairs = VqaDataConfig(path=Path(env("PATHOINFER_VLM_PAIRS")), samples=64)
    cfg = VlmQuantConfig(scheme=scheme, calibration=None if scheme == "fp8" else pairs)
    card = quantise_vlm(model, tmp_path / scheme, cfg)
    assert card.precision == ("fp8" if scheme == "fp8" else "int4")
    assert (scheme == "fp8") == (card.provenance.calibration_set_hash is None)
    reloaded, folder = load_vlm(tmp_path / scheme)
    assert reloaded == card and list(folder.glob("*.safetensors"))
    config = (folder / "config.json").read_text()
    assert "quantization_config" in config and "compressed-tensors" in config


def test_parity_through_vllm_servers(tmp_path: Path) -> None:
    pytest.importorskip("httpx")
    from pathoinfer.convert.vlm_parity import gate

    cfg = VlmParityConfig(eval=VqaDataConfig(path=Path(env("PATHOINFER_VLM_PAIRS")), samples=None))
    report = gate(
        env("PATHOINFER_VLM_MODEL"),
        env("PATHOINFER_VLM_CANDIDATE"),
        cfg,
        tmp_path / "final",
        env("PATHOINFER_VLLM_REFERENCE_URL"),
        env("PATHOINFER_VLLM_CANDIDATE_URL"),
    )
    assert report.samples > 0 and "accuracy_drop" in report.metrics
    assert report.passed == (tmp_path / "final").is_dir()
