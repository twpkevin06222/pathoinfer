"""How to run vLLM for a pathovlm model folder (a ``VqaCard``). Builds settings only;
never imports vllm and never starts a server.

``engine_args`` derives vLLM's engine arguments from the card and the folder's
``config.json``, plus ``VllmServeConfig``:

- ``dtype``: ``bfloat16`` or ``float16`` for bf16/fp16 cards. Quantised cards (fp8,
  int8, int4 from ``pathoinfer convert vlm``) quantise only the language model, so the
  rest runs in ``bfloat16``.
- ``quantization``: the folder's ``quantization_config.quant_method`` (llm-compressor
  writes ``compressed-tensors``). A quantised card without one, or a bf16/fp16 card
  with one, is refused: the card and the weights disagree.
- ``mm_processor_kwargs.max_pixels`` from ``input.max_pixels``, so the server resizes
  crops as in training; ``limit_mm_per_prompt`` one image, as pathoinfer sends.

``serve_command`` turns them into a ``vllm serve`` command line (``pathoinfer serve
vllm``); offline generation (``runtime/local/vllm_offline.py``) passes the same
arguments to ``vllm.LLM``, so both run the model identically.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pathoinfer.registry.model_card import VqaCard

DTYPES = {"bf16": "bfloat16", "fp16": "float16"}
QUANTISED = {"fp8", "int8", "int4"}


class VllmServeConfig(BaseModel):
    """Server sizing. ``max_model_len`` bounds the KV cache per request (prompt with its
    image tokens, plus the answer); the model's own maximum is usually far more than a
    crop question needs and would cost GPU memory. ``allowed_local_media_path`` lets
    clients send ``file://`` crops from under it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    gpu_memory_utilization: float = Field(0.9, gt=0, le=1)
    max_model_len: int = Field(8192, gt=0)
    max_num_seqs: int | None = Field(None, gt=0)
    tensor_parallel_size: int = Field(1, gt=0)
    served_model_name: str | None = None
    allowed_local_media_path: Path | None = None
    host: str = "0.0.0.0"
    port: int = Field(8000, gt=0, lt=65536)


def engine_args(card: VqaCard, folder: Path, cfg: VllmServeConfig) -> dict[str, Any]:
    """vLLM engine arguments (``vllm.EngineArgs`` field names) for the model in ``folder``.

    Raises:
        ValueError: the card's precision and the folder's quantisation disagree.
    """
    args: dict[str, Any] = {
        "model": str(folder),
        "served_model_name": cfg.served_model_name or card.name,
        "dtype": DTYPES.get(card.precision, "bfloat16"),
        "max_model_len": cfg.max_model_len,
        "gpu_memory_utilization": cfg.gpu_memory_utilization,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "limit_mm_per_prompt": {"image": 1},
    }
    method = _quant_method(folder)
    if card.precision in QUANTISED and method is None:
        raise ValueError(
            f"{card.name} is {card.precision} but {folder}/config.json has no quantization_config"
        )
    if card.precision not in QUANTISED and method is not None:
        raise ValueError(
            f"{card.name} is {card.precision} but {folder}/config.json is quantised ({method})"
        )
    if method is not None:
        args["quantization"] = method
    if card.input.max_pixels is not None:
        args["mm_processor_kwargs"] = {"max_pixels": card.input.max_pixels}
    if cfg.max_num_seqs is not None:
        args["max_num_seqs"] = cfg.max_num_seqs
    if cfg.allowed_local_media_path is not None:
        args["allowed_local_media_path"] = str(cfg.allowed_local_media_path.resolve())
    return args


def serve_command(card: VqaCard, folder: Path, cfg: VllmServeConfig) -> list[str]:
    """``vllm serve FOLDER --host ... --port ... --dtype ...`` as an argument list."""
    args = engine_args(card, folder, cfg)
    command = ["vllm", "serve", args.pop("model"), "--host", cfg.host, "--port", str(cfg.port)]
    for key, value in args.items():
        command += [
            f"--{key.replace('_', '-')}",
            json.dumps(value) if isinstance(value, dict) else str(value),
        ]
    return command


def _quant_method(folder: Path) -> str | None:
    config = json.loads((folder / "config.json").read_text())
    quant = config.get("quantization_config")
    return str(quant.get("quant_method", "unknown")) if quant else None
