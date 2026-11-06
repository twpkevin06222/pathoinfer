"""Build TensorRT plans from ONNX model folders, per GPU.

A plan is built as a strongly typed network: its precision is the ONNX graph's (FP32,
FP16 from ``convert fp16``, or INT8/FP8 Q/DQ from ``convert ptq``), so the plan computes
what parity checked. One optimisation profile covers batch sizes ``min_batch`` to
``max_batch``, tuned for ``opt_batch``. Plans run only on the GPU architecture and
TensorRT version they were built with, so built plans are cached under ``cache`` by
(model hash, GPU, compute capability, TensorRT version, precision, profile).
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from pathoinfer._extras import extra, require
from pathoinfer._logging import log_call
from pathoinfer.convert.common import write_variant
from pathoinfer.registry.loader import load_model
from pathoinfer.registry.model_card import ModelCard

CACHE = Path.home() / ".cache" / "pathoinfer" / "plans"


@log_call
def build_plan(
    source: str | Path,
    out: Path,
    *,
    min_batch: int = 1,
    opt_batch: int = 32,
    max_batch: int = 64,
    workspace_gb: float = 4.0,
    cache: Path = CACHE,
    device: int = 0,
) -> ModelCard:
    """Build (or reuse from ``cache``) a plan for the ONNX model at ``source`` on CUDA
    device ``device`` and write it as the new model folder ``out``."""
    trt = require("tensorrt", extra("trt"))
    if not 1 <= min_batch <= opt_batch <= max_batch:
        raise ValueError(
            f"need 1 <= min <= opt <= max batch, got {min_batch}, {opt_batch}, {max_batch}"
        )
    card, model_dir = load_model(str(source))
    if card.framework != "onnx":
        raise ValueError(f"expected an ONNX model, got {card.framework}")
    onnx_bytes = (model_dir / card.file).read_bytes()
    gpu = torch.cuda.get_device_name(device)
    capability = "{}.{}".format(*torch.cuda.get_device_capability(device))
    key_fields = {
        "model": hashlib.sha256(onnx_bytes).hexdigest()[:16],
        "gpu": gpu,
        "compute_capability": capability,
        "tensorrt": trt.__version__,
        "precision": card.precision,
        "profile": [min_batch, opt_batch, max_batch],
    }
    key = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()[:16]
    cached = cache / f"{key}.plan"
    if cached.is_file():
        logger.info("plan cache hit: {}", cached)
    else:
        with torch.cuda.device(device):
            plan = _build(trt, onnx_bytes, card, (min_batch, opt_batch, max_batch), workspace_gb)
        cache.mkdir(parents=True, exist_ok=True)
        partial = cached.with_suffix(".partial")
        partial.write_bytes(plan)
        os.replace(partial, cached)
        logger.info("built plan {} ({:.1f} MB)", cached, len(plan) / 1e6)
    tag = f"{gpu.replace(' ', '-').lower()}-sm{capability.replace('.', '')}"
    return write_variant(
        card,
        out,
        lambda path: shutil.copyfile(cached, path),
        framework="tensorrt",
        file="model.plan",
        version=f"{card.version}+trt-{tag}",
        **{
            "provenance.source_metadata": {
                **card.provenance.source_metadata,
                "tensorrt": key_fields | {"cache_key": key},
            }
        },
    )


def _build(
    trt: Any, onnx_bytes: bytes, card: ModelCard, batches: tuple[int, int, int], workspace_gb: float
) -> bytes:
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, log)
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise ValueError("TensorRT could not parse the ONNX model:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 2**30))
    profile = builder.create_optimization_profile()
    size = card.input.tile_size
    shape = (3, size, size) if card.input.layout == "NCHW" else (size, size, 3)
    profile.set_shape(card.input.name, *[(b, *shape) for b in batches])
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("TensorRT failed to build the plan (see its log above)")
    return bytes(plan)
