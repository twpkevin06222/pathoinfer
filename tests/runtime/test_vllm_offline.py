"""Offline vLLM generation. vllm is not installed in the main venv: these tests check the
module refuses without it, and drive it through a stand-in ``vllm`` module."""

import importlib.util
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from pathoinfer.registry.loader import load_vlm
from pathoinfer.runtime.local.vllm_offline import OfflineVlm
from pathoinfer.runtime.vlm import chat_messages, image_url
from pathoinfer.serving.vllm import VllmServeConfig, engine_args


@pytest.mark.skipif(importlib.util.find_spec("vllm") is not None, reason="vllm is installed")
def test_refuses_without_vllm_and_names_the_separate_venv(vqa_folder: Callable[..., Path]) -> None:
    with pytest.raises(ImportError, match="separate Python 3.13 venv"):
        OfflineVlm(vqa_folder())


def fake_vllm(calls: dict[str, Any]) -> types.ModuleType:
    module = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

    class LLM:
        def __init__(self, **kwargs: Any):
            calls["engine"] = kwargs

        def chat(self, conversations: list[Any], params: Any, use_tqdm: bool) -> list[Any]:
            calls["chat"] = (conversations, params.kwargs)
            text = [c[-1]["content"][1]["text"].upper() for c in conversations]
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=t)]) for t in text]

    module.LLM, module.SamplingParams = LLM, SamplingParams  # type: ignore[attr-defined]
    return module


def test_generates_with_the_servers_engine_args_and_the_clients_messages(
    vqa_folder: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: dict[str, Any] = {}
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm(calls))
    folder = vqa_folder(max_pixels=448 * 448)
    path = tmp_path / "crop.png"
    cv2.imwrite(str(path), np.zeros((16, 16, 3), np.uint8))

    vlm = OfflineVlm(folder, VllmServeConfig(max_model_len=4096))
    answers = vlm.generate([(path, "tumour?"), (path, "necrosis?")], max_tokens=8)

    assert answers == ["TUMOUR?", "NECROSIS?"]
    card, model_dir = load_vlm(folder)
    assert calls["engine"] == engine_args(card, model_dir, VllmServeConfig(max_model_len=4096))
    conversations, params = calls["chat"]
    assert conversations[0] == chat_messages(image_url(path), "tumour?")
    assert params == {"temperature": 0, "max_tokens": 8}
