import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pathoinfer.registry.model_card import ModelCard, parse_card
from pathoinfer.runtime.base import Outputs, preprocess, run_in_fixed_batches

CARD = json.loads((Path(__file__).parents[1] / "fixtures/tiny_seg/model_card.json").read_text())
MEAN, STD, SCALE = (0.1, 0.5, 0.9), (0.2, 0.25, 0.5), 1 / 100


def card(**input_changes: object) -> ModelCard:
    normalisation = {"mean": MEAN, "std": STD, "scale": SCALE}
    return parse_card(
        CARD | {"input": CARD["input"] | {"normalisation": normalisation} | input_changes}
    )


def tiles(n: int = 3) -> torch.Tensor:
    return torch.from_numpy(np.random.default_rng(0).integers(0, 256, (n, 16, 16, 3), np.uint8))


def expected(batch: torch.Tensor, order: str, layout: str) -> np.ndarray:
    x = batch.numpy().astype(np.float64)
    if order == "BGR":
        x = x[..., ::-1]
    x = (x * SCALE - np.array(MEAN)) / np.array(STD)
    return x.transpose(0, 3, 1, 2) if layout == "NCHW" else x


@pytest.mark.parametrize("order", ["RGB", "BGR"])
@pytest.mark.parametrize("layout", ["NCHW", "NHWC"])
def test_preprocess_applies_the_cards_numbers(order: str, layout: str) -> None:
    batch = tiles()
    x = preprocess(batch, card(channel_order=order, layout=layout), torch.device("cpu"))
    assert x.dtype == torch.float32 and x.is_contiguous()
    np.testing.assert_allclose(x.numpy(), expected(batch, order, layout), rtol=1e-6, atol=1e-6)


def test_preprocess_casts_to_float16() -> None:
    batch = tiles()
    x = preprocess(batch, card(dtype="float16"), torch.device("cpu"))
    assert x.dtype == torch.float16
    np.testing.assert_allclose(x.float().numpy(), expected(batch, "RGB", "NCHW"), atol=5e-3)


def test_uint8_input_skips_normalisation() -> None:
    batch = tiles()
    x = preprocess(batch, card(dtype="uint8", channel_order="BGR"), torch.device("cpu"))
    assert x.dtype == torch.uint8
    assert torch.equal(x, batch.flip(-1).permute(0, 3, 1, 2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA GPU")
def test_preprocess_on_gpu_matches_cpu() -> None:
    batch = tiles().pin_memory()
    gpu = preprocess(batch, card(), torch.device("cuda", 0))
    assert gpu.device.type == "cuda"
    torch.testing.assert_close(gpu.cpu(), preprocess(batch, card(), torch.device("cpu")))


@pytest.mark.parametrize("n", [1, 4, 5, 9])
def test_fixed_batches_pad_the_last_chunk_and_strip_the_padding(n: int) -> None:
    sizes = []

    def run(chunk: torch.Tensor) -> Outputs:
        sizes.append(chunk.shape[0])
        return {"sum": chunk.flatten(1).sum(1), "first": chunk[:, 0, 0, 0]}

    batch = tiles(n)
    out = run_in_fixed_batches(batch, 4, run)
    assert set(sizes) == {4} and len(sizes) == -(-n // 4)
    assert torch.equal(out["sum"], batch.flatten(1).sum(1))
    assert torch.equal(out["first"], batch[:, 0, 0, 0])


def test_empty_batch_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        run_in_fixed_batches(tiles(0), 4, lambda chunk: {})
