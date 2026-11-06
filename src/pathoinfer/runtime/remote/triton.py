"""Triton Inference Server gRPC client runtime, for tile models.

``TritonRuntime`` implements the ``Runtime`` protocol, so the engine runs a slide the same
way against a server as in-process: tiles are preprocessed here exactly as by the local
runtimes (``runtime.base.preprocess``, driven by the ModelCard), sent as one gRPC request
per fixed-size batch, and the outputs come back as CPU tensors keyed by the card's
output names. The model must be served under ``model_name`` (default: the card's name),
e.g. from a repository written by ``serving.triton_repo``.
"""

from typing import Any

import numpy as np
import torch

from pathoinfer._extras import extra, require
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Outputs, preprocess, run_in_fixed_batches

TRITON_DTYPES = {"float32": "FP32", "float16": "FP16", "uint8": "UINT8"}


class TritonRuntime:
    def __init__(
        self,
        card: ModelCard,
        url: str,
        model_name: str | None = None,
        batch_size: int = 32,
        timeout_s: float = 60.0,
    ):
        """``url`` is Triton's gRPC address, e.g. ``triton:8001``.

        Raises:
            ValueError: the server does not serve a ready ``model_name``, or its input
                does not match the card.
        """
        self._grpc = require("tritonclient.grpc", extra("triton"))
        self.card, self.batch_size = card, batch_size
        self.device = torch.device("cpu")
        self.model = model_name or card.name
        self.timeout_s = timeout_s
        self.client = self._grpc.InferenceServerClient(url)
        if not self.client.is_model_ready(self.model):
            raise ValueError(f"Triton at {url} has no ready model {self.model!r}")
        metadata = self.client.get_model_metadata(self.model, as_json=True)
        inputs = {i["name"]: i["datatype"] for i in metadata["inputs"]}
        wanted = TRITON_DTYPES[card.input.dtype]
        if inputs.get(card.input.name) != wanted:
            raise ValueError(
                f"{self.model}: Triton input {inputs}; the card wants {card.input.name!r} {wanted}"
            )
        self._outputs = card.output.tensor_names()

    def predict(self, batch: torch.Tensor) -> Outputs:
        return run_in_fixed_batches(batch, self.batch_size, self._run)

    def warmup(self) -> None:
        size = self.card.input.tile_size
        self.predict(torch.zeros((self.batch_size, size, size, 3), dtype=torch.uint8))

    def close(self) -> None:
        self.client.close()

    def _run(self, chunk: torch.Tensor) -> Outputs:
        x = np.ascontiguousarray(preprocess(chunk, self.card, self.device).numpy())
        request = self._grpc.InferInput(
            self.card.input.name, list(x.shape), TRITON_DTYPES[str(x.dtype)]
        )
        request.set_data_from_numpy(x)
        result: Any = self.client.infer(
            self.model,
            [request],
            outputs=[self._grpc.InferRequestedOutput(name) for name in self._outputs],
            client_timeout=self.timeout_s,
        )
        return {name: torch.from_numpy(np.array(result.as_numpy(name))) for name in self._outputs}
