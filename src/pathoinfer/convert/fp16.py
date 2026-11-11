"""ONNX FP32 -> FP16 with ONNX Runtime's converter.

Weights and activations become float16 except for blocked op types, which stay float32
with Casts around them. ``keep_io_types`` keeps the graph's inputs and outputs float32,
so preprocessing and stitchers see the same dtypes as before; without it the card's
``input.dtype`` becomes ``float16``.

Models over 2 GB keep their weights in a file beside the graph. Protobuf cannot hold
them in memory as one message, so shapes are inferred from the file
(``infer_shapes_path``) and an FP16 model over ``EXTERNAL_BYTES`` is saved with its
weights in ``<file>.data``.

Weights above FP16's range would be clipped to 65504 by the converter, silently: HoVer-Net's
decoder BatchNorms hold running variances up to 3e7, and clipping them scaled their
outputs up to 22x (parity F1 0). So a BatchNorm whose parameters do not fit is first
rewritten to the same affine map in a form that does (``scale = gamma / sqrt(var + eps)``
as gamma, ``beta - mean * scale`` as beta, mean 0, variance 1, epsilon 0), and any other
out-of-range weight feeding an FP16 op raises, naming it.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from pathoinfer._extras import extra, require
from pathoinfer._logging import log_call
from pathoinfer.convert.common import write_variant
from pathoinfer.registry.loader import load_model
from pathoinfer.registry.model_card import ModelCard

EXTERNAL_BYTES = 2**30  # well under protobuf's 2 GB message limit
FP16_MAX = float(np.finfo(np.float16).max)


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
    with tempfile.TemporaryDirectory() as tmp:  # the converter's own inference is in memory
        inferred = Path(tmp) / "inferred.onnx"
        onnx.shape_inference.infer_shapes_path(str(model_dir / card.file), str(inferred))
        model = onnx.load(inferred, load_external_data=False)
    onnx.load_external_data_for_model(model, str(model_dir))
    if "BatchNormalization" not in blocked:
        fit_batchnorms(model)
    check_range(model, set(blocked))
    model = float16.convert_float_to_float16(
        model, keep_io_types=keep_io_types, op_block_list=blocked, disable_shape_infer=True
    )
    external = sum(len(t.raw_data) for t in model.graph.initializer) > EXTERNAL_BYTES

    def save(path: Path) -> None:
        if external:
            onnx.save(
                model,
                path,
                save_as_external_data=True,
                location=path.name + ".data",
                size_threshold=0,
            )
        else:
            onnx.save(model, path)

    updates: dict[str, object] = {"precision": "fp16", "version": card.version + version_suffix}
    if not keep_io_types:
        updates["input.dtype"] = "float16"
    return write_variant(card, out, save, **updates)


def fit_batchnorms(model: Any) -> None:
    """Rewrite, in place, each BatchNorm whose parameters exceed FP16's range to the same
    affine map with parameters that fit (see the module docstring)."""
    from onnx import helper, numpy_helper

    init = {i.name: i for i in model.graph.initializer}
    for node in model.graph.node:
        params = node.input[1:5]
        if node.op_type != "BatchNormalization" or not all(n in init for n in params):
            continue
        gamma, beta, mean, var = (numpy_helper.to_array(init[n]).astype(np.float64) for n in params)
        if max(np.abs(a).max() for a in (gamma, beta, mean, var)) <= FP16_MAX:
            continue
        eps = next((a.f for a in node.attribute if a.name == "epsilon"), 1e-5)
        scale = gamma / np.sqrt(var + eps)
        values = (scale, beta - mean * scale, np.zeros_like(mean), np.ones_like(var))
        dtype = numpy_helper.to_array(init[params[0]]).dtype
        names = [f"{n}_fp16_fit" for n in params]  # new names: the originals may be shared
        model.graph.initializer.extend(
            numpy_helper.from_array(v.astype(dtype), n) for v, n in zip(values, names, strict=True)
        )
        node.input[1:5] = names
        kept = [a for a in node.attribute if a.name != "epsilon"]
        del node.attribute[:]
        node.attribute.extend([*kept, helper.make_attribute("epsilon", 0.0)])
    used = {i for node in model.graph.node for i in node.input}
    unused = [i for i in model.graph.initializer if i.name not in used]
    for i in unused:
        model.graph.initializer.remove(i)


def check_range(model: Any, blocked: set[str]) -> None:
    """Raise if a float weight outside FP16's range feeds an op that becomes FP16."""
    from onnx import numpy_helper

    consumers: dict[str, set[str]] = {}
    for node in model.graph.node:
        for i in node.input:
            consumers.setdefault(i, set()).add(node.op_type)
    bad = []
    for i in model.graph.initializer:
        ops = consumers.get(i.name, set()) - blocked
        a = numpy_helper.to_array(i)
        if ops and a.dtype.kind == "f" and a.size and np.abs(a).max() > FP16_MAX:
            bad.append(f"{i.name} ({', '.join(sorted(ops))}, max {np.abs(a).max():.3g})")
    if bad:
        raise ValueError(
            f"{len(bad)} weights exceed FP16's range ({FP16_MAX:g}) and would be clipped: "
            + "; ".join(bad[:5])
            + ". Keep their op types in FP32 with --keep-fp32."
        )
