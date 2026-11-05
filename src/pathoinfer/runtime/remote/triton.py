"""Triton Inference Server gRPC client runtime. Lands in v0.3."""

from pathoinfer._extras import extra, require
from pathoinfer.registry.model_card import ModelCard
from pathoinfer.runtime.base import Runtime


def connect(card: ModelCard, url: str) -> Runtime:
    require("tritonclient.grpc", extra("triton"))
    raise NotImplementedError("lands in v0.3")
