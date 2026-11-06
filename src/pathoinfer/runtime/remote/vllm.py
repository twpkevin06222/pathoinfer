"""OpenAI-compatible HTTP client to a vLLM server, for VLMs. Never imports the vllm
package: start the server with ``vllm serve MODEL_FOLDER`` and point this at its URL.

``VllmClient.ask`` sends one slide crop and a question and returns the answer text,
greedy (temperature 0) so the answers are reproducible. The full Runtime for VLM
inference over slides builds on this later.
"""

from pathlib import Path
from typing import Any

from pathoinfer._extras import extra, require


class VllmClient:
    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        timeout_s: float = 120.0,
        transport: Any = None,
    ):
        """``base_url`` is the server root, e.g. ``http://localhost:8000``; ``model`` the
        served model name (default: the server's first). ``transport`` replaces the HTTP
        transport (tests use ``httpx.MockTransport``)."""
        httpx = require("httpx", extra("vllm-client"))
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport
        )
        self.model = model or self._first_model()

    def ask(self, image: str | Path, question: str, max_tokens: int = 64) -> str:
        from pathoinfer.convert.vlm import image_data_url

        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(image)}},
            {"type": "text", "text": question},
        ]
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        response = self.http.post("/v1/chat/completions", json=body)
        response.raise_for_status()
        answer: str = response.json()["choices"][0]["message"]["content"]
        return answer

    def close(self) -> None:
        self.http.close()

    def _first_model(self) -> str:
        response = self.http.get("/v1/models")
        response.raise_for_status()
        models = response.json()["data"]
        if not models:
            raise ValueError("the server serves no model")
        name: str = models[0]["id"]
        return name
