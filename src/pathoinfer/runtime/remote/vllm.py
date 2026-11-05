"""OpenAI-compatible HTTP client to a vLLM server, for VLMs. Never imports the vllm package.
Lands after v0.3.
"""

from pathoinfer._extras import extra, require
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Runtime


def connect(card: ModelCard, base_url: str) -> Runtime:
    require("httpx", extra("vllm-client"))
    raise NotImplementedError("lands after v0.3")
