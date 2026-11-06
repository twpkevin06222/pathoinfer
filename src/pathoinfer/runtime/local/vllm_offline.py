"""Offline batch generation with vLLM's Python API, for evaluation and captioning.

This is the only pathoinfer module that imports ``vllm``, and only when ``OfflineVlm``
is created. vLLM pins its own torch, numpy and transformers (0.33: torch 2.14.0,
numpy<2.5, transformers<5.21) and clashes with ``[vlm-convert]``, so install it in a
separate venv, Python 3.13+ (``runtime/README.md``). From ``packages/pathoinfer``::

    UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --python 3.13 --extra vllm

For serving, run ``vllm serve`` (``pathoinfer serve vllm``) and use the HTTP client in
``runtime/remote/vllm.py`` instead: it needs no vllm install.

The engine gets the same arguments as the server (``serving.vllm.engine_args``) and the
same chat message as the client (``runtime.vlm.chat_messages``), and decodes greedily, so
offline answers match a server's for the same model.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from pathoinfer._extras import require
from pathoinfer.registry.loader import load_vlm
from pathoinfer.runtime.vlm import Image, chat_messages, image_url
from pathoinfer.serving.vllm import VllmServeConfig, engine_args

INSTALL = (
    "install it in a separate Python 3.13 venv: "
    "UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --python 3.13 --extra vllm "
    "(see pathoinfer's runtime/README.md)"
)


class OfflineVlm:
    def __init__(self, source: str | Path, cfg: VllmServeConfig | None = None):
        """Load the VLM folder ``source`` (folder, ``models:/``, ``hf://``) into a vLLM
        engine. ``cfg``'s server settings (host, port) are ignored."""
        self._vllm = require("vllm", INSTALL)
        self.card, self.folder = load_vlm(source)
        args = engine_args(self.card, self.folder, cfg or VllmServeConfig())
        logger.info("vLLM engine for {} {}: {}", self.card.name, self.card.precision, args)
        self.llm = self._vllm.LLM(**args)

    def generate(
        self, items: Sequence[tuple[Image, str]], max_tokens: int = 64, *, system: str | None = None
    ) -> list[str]:
        """The answer to each ``(image, question)``, in order, greedy. vLLM batches the
        whole list itself."""
        conversations: list[Any] = [
            chat_messages(image_url(image), question, system) for image, question in items
        ]
        params = self._vllm.SamplingParams(temperature=0, max_tokens=max_tokens)
        outputs = self.llm.chat(conversations, params, use_tqdm=False)
        return [str(output.outputs[0].text) for output in outputs]
