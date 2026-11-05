from pathlib import Path

import numpy as np
import pytest
import torch

from pathoinfer.config import RuntimeConfig
from pathoinfer.registry.loader import load_model
from pathoinfer.runtime.base import Runtime
from pathoinfer.runtime.factory import make_runtime
from pathoinfer.runtime.local.torch import TorchRuntime

FIXTURES = Path(__file__).parents[1] / "fixtures"
CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="no CUDA GPU")


def runtime(folder: str, batch_size: int = 4, **config: object) -> Runtime:
    card, model_dir = load_model(FIXTURES / folder)
    return make_runtime(card, model_dir, RuntimeConfig(batch_size=batch_size, **config))  # type: ignore[arg-type]


def tiles(n: int = 5) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(1).integers(0, 256, (n, 16, 16, 3), np.uint8))


@pytest.fixture(scope="module")
def reference() -> np.ndarray:
    pytest.importorskip("onnxruntime")
    return runtime("tiny_seg", device="cpu").predict(tiles())["logits"].numpy()


# (folder, config, tolerance); every runtime gets the same 5 tiles in batches of 4.
CASES: dict[str, tuple[str, dict[str, object], float]] = {
    "onnx-cpu": ("tiny_seg", {"device": "cpu"}, 1e-5),
    "onnx-cuda": ("tiny_seg", {"device": "cuda", "tensorrt": False}, 1e-4),
    "onnx-fp16-cuda": ("tiny_seg_fp16", {"device": "cuda", "tensorrt": False}, 2e-2),
    "torchscript-cpu": ("tiny_seg_torch", {"device": "cpu"}, 1e-5),
    "torchscript-cuda": ("tiny_seg_torch", {"device": "cuda"}, 1e-4),
    "torchscript-cuda-bf16": ("tiny_seg_torch", {"device": "cuda", "autocast": "bf16"}, 1e-1),
    "torchscript-cuda-fp16": ("tiny_seg_torch", {"device": "cuda", "autocast": "fp16"}, 2e-2),
    "export-cpu": ("tiny_seg_export", {"device": "cpu"}, 1e-5),
    "export-cuda": ("tiny_seg_export", {"device": "cuda"}, 1e-4),
}


@pytest.mark.parametrize("name", CASES)
def test_every_runtime_matches_onnx_on_cpu(name: str, reference: np.ndarray) -> None:
    folder, config, tolerance = CASES[name]
    if config.get("device") == "cuda" and not CUDA:
        pytest.skip("no CUDA GPU")
    rt = runtime(folder, **config)
    rt.warmup()
    out = rt.predict(tiles())["logits"]
    assert out.shape == (5, 2, 16, 16)
    assert out.device.type == rt.device.type == config["device"]
    np.testing.assert_allclose(out.float().cpu().numpy(), reference, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("folder", ["tiny_seg", "tiny_seg_torch"])
def test_padding_does_not_change_real_outputs(folder: str) -> None:
    if folder == "tiny_seg":
        pytest.importorskip("onnxruntime")
    batch = tiles(5)
    whole = runtime(folder, batch_size=4, device="cpu").predict(batch)["logits"]
    one_by_one = runtime(folder, batch_size=1, device="cpu").predict(batch)["logits"]
    torch.testing.assert_close(whole, one_by_one)


@needs_cuda
def test_onnx_on_cuda_keeps_outputs_on_the_gpu() -> None:
    pytest.importorskip("onnxruntime")
    rt = runtime("tiny_seg", device="cuda", tensorrt=False)
    assert "CUDAExecutionProvider" in rt.providers  # type: ignore[attr-defined]
    assert rt.predict(tiles().pin_memory())["logits"].device.type == "cuda"


def test_card_dtype_must_match_the_graph() -> None:
    pytest.importorskip("onnxruntime")
    from pathoinfer.runtime.local.onnx import OnnxRuntime

    card, _ = load_model(FIXTURES / "tiny_seg_fp16")
    with pytest.raises(ValueError, match="float16, graph has float32"):
        OnnxRuntime(card, FIXTURES / "tiny_seg" / "model.onnx", device="cpu")


def test_factory_rejects_weights_only_models() -> None:
    card, model_dir = load_model(FIXTURES / "tiny_seg")
    other = card.model_copy(update={"framework": "safetensors", "file": "model.safetensors"})
    with pytest.raises(ValueError, match="weights only"):
        make_runtime(other, model_dir, RuntimeConfig())


def test_factory_builds_a_torch_runtime_for_torch_cards() -> None:
    assert isinstance(runtime("tiny_seg_torch", device="cpu"), TorchRuntime)
