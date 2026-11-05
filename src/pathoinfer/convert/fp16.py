"""ONNX FP32 -> FP16 with ONNX Runtime's converter.

Weights and activations become float16 except for blocked op types, which stay float32
with Casts around them. ``keep_io_types`` keeps the graph's inputs and outputs float32,
so preprocessing and stitchers see the same dtypes as before; without it the card's
``input.dtype`` becomes ``float16``.
"""

from pathlib import Path

from pathoinfer._extras import extra, require
from pathoinfer._logging import log_call
from pathoinfer.convert.common import write_variant
from pathoinfer.registry.loader import load_model
from pathoinfer.registry.model_card import ModelCard


@log_call
def to_fp16(
    source: str | Path,
    out: Path,
    *,
    keep_io_types: bool = True,
    keep_fp32_ops: list[str] | None = None,
    version_suffix: str = "+fp16",
) -> ModelCard:
    """Convert the ONNX model at ``source`` (any model source) to FP16 in the new folder
    ``out``. ``keep_fp32_ops`` adds op types to ONNX Runtime's default block list."""
    onnx = require("onnx", extra("convert"))
    float16 = require("onnxruntime.transformers.float16", extra("onnx"))
    card, model_dir = load_model(str(source))
    if card.framework != "onnx" or card.precision != "fp32":
        raise ValueError(f"expected an fp32 ONNX model, got {card.framework} {card.precision}")
    blocked = sorted({*float16.DEFAULT_OP_BLOCK_LIST, *(keep_fp32_ops or [])})
    model = float16.convert_float_to_float16(
        onnx.load(model_dir / card.file), keep_io_types=keep_io_types, op_block_list=blocked
    )
    updates: dict[str, object] = {"precision": "fp16", "version": card.version + version_suffix}
    if not keep_io_types:
        updates["input.dtype"] = "float16"
    return write_variant(card, out, lambda path: onnx.save(model, path), **updates)
