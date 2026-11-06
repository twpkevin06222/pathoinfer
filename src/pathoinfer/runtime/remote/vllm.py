"""OpenAI-compatible HTTP client to a vLLM server, for VLMs. Never imports the vllm
package: start the server with ``vllm serve MODEL_FOLDER`` (``serving/vllm.py`` builds
the command from the model card) and point this at its URL.

- ``ask`` sends one image and a question and returns the answer text, greedy
  (temperature 0) so answers are reproducible. ``json_schema`` constrains the answer
  to a JSON schema (vLLM's structured outputs).
- ``ask_many`` asks a list of questions concurrently and returns answers in order.
- Images go as base64 data URLs, or as ``file://`` URIs with ``local_media=True`` when
  the server runs with ``--allowed-local-media-path`` and sees the same files.
- At most ``concurrency`` requests are in flight per client, whichever threads send
  them. Connection errors, timeouts and HTTP 408/429/5xx are retried ``retries`` times
  with exponential backoff; other errors raise ``httpx.HTTPStatusError`` at once.

Boxes in answers become level-0 coordinates with ``runtime.vlm.parse_boxes``.
"""

import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from loguru import logger

from pathoinfer._extras import extra, require
from pathoinfer.runtime.vlm import Image, chat_messages, image_url

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class VllmClient:
    def __init__(
        self,
        base_url: str,
        model: str | None = None,
        *,
        timeout_s: float = 120.0,
        retries: int = 3,
        backoff_s: float = 1.0,
        concurrency: int = 8,
        local_media: bool = False,
        api_key: str | None = None,
        transport: Any = None,
    ):
        """``base_url`` is the server root, e.g. ``http://localhost:8000``; ``model`` the
        served model name (default: the server's first). ``transport`` replaces the HTTP
        transport (tests use ``httpx.MockTransport``)."""
        if concurrency < 1 or retries < 0:
            raise ValueError("concurrency must be >= 1 and retries >= 0")
        self._httpx = require("httpx", extra("vllm-client"))
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = self._httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_s,
            headers=headers,
            limits=self._httpx.Limits(max_connections=concurrency),
            transport=transport,
        )
        self.retries, self.backoff_s = retries, backoff_s
        self.concurrency, self.local_media = concurrency, local_media
        self._slots = threading.BoundedSemaphore(concurrency)
        self.model = model or self._first_model()

    def ask(
        self,
        image: Image,
        question: str,
        max_tokens: int = 64,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        url = image_url(image, self.local_media)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages(url, question, system),
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if json_schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": json_schema},
            }
        answer: str = self._request("POST", "/v1/chat/completions", json=body)["choices"][0][
            "message"
        ]["content"]
        return answer

    def ask_many(
        self, items: Sequence[tuple[Image, str]], max_tokens: int = 64, **kwargs: Any
    ) -> list[str]:
        """``ask`` for each ``(image, question)``, ``concurrency`` at a time, in order."""
        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(
                pool.map(
                    lambda item: self.ask(item[0], item[1], max_tokens=max_tokens, **kwargs), items
                )
            )

    def close(self) -> None:
        self.http.close()

    def _first_model(self) -> str:
        models = self._request("GET", "/v1/models")["data"]
        if not models:
            raise ValueError("the server serves no model")
        name: str = models[0]["id"]
        return name

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        for attempt in range(self.retries + 1):
            try:
                with self._slots:
                    response = self.http.request(method, path, **kwargs)
            except self._httpx.TransportError as e:  # connection errors and timeouts
                if attempt == self.retries:
                    raise
                reason = type(e).__name__
            else:
                if response.status_code not in RETRY_STATUS or attempt == self.retries:
                    response.raise_for_status()
                    return response.json()
                reason = f"HTTP {response.status_code}"
            delay = self.backoff_s * 2**attempt
            logger.warning(
                "vLLM {} {} failed ({}); retry {}/{} in {:.1f} s",
                method,
                path,
                reason,
                attempt + 1,
                self.retries,
                delay,
            )
            time.sleep(delay)
        raise AssertionError("unreachable")
