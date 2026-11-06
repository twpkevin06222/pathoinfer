"""VllmClient against a mocked OpenAI-compatible server (httpx.MockTransport)."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

httpx = pytest.importorskip("httpx")

from pathoinfer.runtime.remote.vllm import VllmClient


class Server:
    """Serves one model; answers each chat request with its question, after any
    scripted failures (status codes, or exceptions to raise)."""

    def __init__(self, failures: list[Any] | None = None, delay_s: float = 0.0):
        self.failures = list(failures or [])
        self.delay_s = delay_s
        self.requests: list[dict[str, Any]] = []
        self.headers: list[Any] = []
        self.in_flight = self.max_in_flight = 0
        self.lock = threading.Lock()

    def __call__(self, request: Any) -> Any:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "path-vqa"}]})
        with self.lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            failure = self.failures.pop(0) if self.failures else None
            body = json.loads(request.content)
            self.requests.append(body)
            self.headers.append(request.headers)
        try:
            time.sleep(self.delay_s)
            if isinstance(failure, Exception):
                raise failure
            if failure is not None:
                return httpx.Response(failure, json={"error": "scripted"})
            question = body["messages"][-1]["content"][1]["text"]
            return httpx.Response(200, json={"choices": [{"message": {"content": question}}]})
        finally:
            with self.lock:
                self.in_flight -= 1


def client(server: Server, **kwargs: Any) -> VllmClient:
    return VllmClient(
        "http://server/", transport=httpx.MockTransport(server), backoff_s=0, **kwargs
    )


def crop(tmp_path: Path) -> Path:
    path = tmp_path / "crop.png"
    cv2.imwrite(str(path), np.zeros((16, 16, 3), np.uint8))
    return path


def test_sends_the_image_question_and_greedy_settings(tmp_path: Path) -> None:
    server = Server()
    c = client(server, api_key="secret")
    assert c.model == "path-vqa"
    assert c.ask(crop(tmp_path), "tumour?", max_tokens=16) == "tumour?"
    body = server.requests[0]
    assert body["model"] == "path-vqa" and body["temperature"] == 0 and body["max_tokens"] == 16
    content = body["messages"][0]["content"]
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1] == {"type": "text", "text": "tumour?"}
    assert "response_format" not in body
    assert server.headers[0]["authorization"] == "Bearer secret"


def test_local_media_sends_file_uris_and_json_schema_constrains_the_answer(tmp_path: Path) -> None:
    server = Server()
    schema = {"type": "array", "items": {"type": "object"}}
    client(server, local_media=True).ask(crop(tmp_path), "find tumour", json_schema=schema)
    body = server.requests[0]
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    assert url == (tmp_path / "crop.png").resolve().as_uri()
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "answer", "schema": schema},
    }


def test_server_errors_and_timeouts_are_retried(tmp_path: Path) -> None:
    server = Server(failures=[503, httpx.ReadTimeout("slow"), 429])
    assert client(server, retries=3).ask(crop(tmp_path), "tumour?") == "tumour?"
    assert len(server.requests) == 4


def test_retries_run_out(tmp_path: Path) -> None:
    server = Server(failures=[503] * 3)
    with pytest.raises(httpx.HTTPStatusError, match="503"):
        client(server, retries=2).ask(crop(tmp_path), "tumour?")
    assert len(server.requests) == 3
    server = Server(failures=[httpx.ConnectError("down")] * 2)
    with pytest.raises(httpx.ConnectError):
        client(server, retries=1).ask(crop(tmp_path), "tumour?")


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    server = Server(failures=[400])
    with pytest.raises(httpx.HTTPStatusError, match="400"):
        client(server).ask(crop(tmp_path), "tumour?")
    assert len(server.requests) == 1


def test_ask_many_keeps_order_and_runs_concurrently(tmp_path: Path) -> None:
    server = Server(delay_s=0.02)
    items = [(crop(tmp_path), f"q{i}") for i in range(12)]
    assert client(server, concurrency=4).ask_many(items) == [f"q{i}" for i in range(12)]
    assert 1 < server.max_in_flight <= 4


def test_concurrency_limit_holds_across_callers_threads(tmp_path: Path) -> None:
    server = Server(delay_s=0.02)
    c, image = client(server, concurrency=2), crop(tmp_path)
    with ThreadPoolExecutor(10) as pool:
        answers = list(pool.map(lambda i: c.ask(image, f"q{i}"), range(10)))
    assert answers == [f"q{i}" for i in range(10)]
    assert server.max_in_flight == 2


def test_arrays_are_sent_as_png() -> None:
    server = Server()
    client(server).ask(np.zeros((8, 8, 3), np.uint8), "tumour?")
    url = server.requests[0]["messages"][0]["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
